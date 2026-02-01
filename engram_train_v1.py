"""
================================================================================
[Engram Architecture 8-GPU DDP Training Demo]

Environment: 8x NVIDIA A100-SXM4-80GB
Modifications for DistributedDataParallel (DDP):
1. Added 'torch.distributed' setup and cleanup.
2. Wrapped model with 'DistributedDataParallel'.
3. Implemented 'SyntheticDataset' and used 'DistributedSampler'.
4. Added rank-aware printing (only Rank 0 prints logs).
================================================================================
"""

import os
import math
import random
from typing import List
from dataclasses import dataclass, field
from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex 

# ==========================================
# 1. Configuration
# ==========================================
@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "/data2/home/wanghaoyi/models/gpt2_local" 
    engram_vocab_size: List[int] = field(default_factory=lambda: [10000, 10000]) 
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 128
    n_head_per_ngram: int = 4
    layer_ids: List[int] = field(default_factory=lambda: [0, 1])
    pad_id: int = 50256
    seed: int = 42
    kernel_size: int = 4
    
@dataclass
class BackBoneConfig:
    hidden_size: int = 128
    hc_mult: int = 4
    vocab_size: int = 50257
    num_layers: int = 2
    
engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()

# ==========================================
# 2. Tokenizer & Hashing Utilities
# ==========================================
class CompressedTokenizer:
    def __init__(self, tokenizer_name_or_path):
        # In DDP, ensure only local_rank 0 downloads, others wait
        # ideally, model should be pre-downloaded. 
        # Here we assume internet access or cached model.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.lookup_table, self.num_new_token = self._build_lookup_table()
    
    def __len__(self):
        return self.num_new_token
    
    def _build_lookup_table(self):
        vocab_size = len(self.tokenizer)
        lookup = np.arange(vocab_size, dtype=np.int64)
        return lookup, vocab_size
    
    def _compress(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        arr = np.clip(arr, 0, len(self.lookup_table)-1)
        return self.lookup_table[arr]
    
    def __call__(self, input_ids):
        return self._compress(input_ids)

def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1

class NgramHashMapping:
    def __init__(self, cfg: EngramConfig):
        self.vocab_size_per_ngram = cfg.engram_vocab_size
        self.max_ngram_size = cfg.max_ngram_size
        self.n_head_per_ngram = cfg.n_head_per_ngram
        self.pad_id = cfg.pad_id
        self.layer_ids = cfg.layer_ids

        self.compressed_tokenizer = CompressedTokenizer(cfg.tokenizer_name_or_path)            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        
        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        self.layer_multipliers = {}
        # Important: Deterministic seed ensures all GPUs generate SAME hash mapping
        for layer_id in self.layer_ids:
            base_seed = int(cfg.seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(0, half_bound, size=(self.max_ngram_size,), dtype=np.int64)
            self.layer_multipliers[layer_id] = r * 2 + 1

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                current_prime_search_start = vocab_size - 1
                for _ in range(self.n_head_per_ngram):
                    found_prime = find_next_prime(current_prime_search_start, seen_primes)
                    seen_primes.add(found_prime)
                    current_ngram_heads_sizes.append(found_prime)
                    current_prime_search_start = found_prime
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
        return vocab_size_across_layers

    def _get_ngram_hashes(self, input_ids: np.ndarray, layer_id: int) -> np.ndarray:
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape
        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0: return x
            shifted = np.pad(x, ((0, 0), (k, 0)), mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]
        all_hashes = []
        
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = (tokens[0] * multipliers[0])
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            for j in range(self.n_head_per_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids_numpy):
        compressed_ids = self.compressed_tokenizer(input_ids_numpy)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(compressed_ids, layer_id=layer_id)
        return hash_ids_for_all_layers

# ==========================================
# 3. Model Components
# ==========================================
class ShortConv(nn.Module):
    def __init__(self, hidden_size, kernel_size, dilation, hc_mult):
        super().__init__()
        self.hc_mult = hc_mult
        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.norms = nn.ModuleList([nn.RMSNorm(hidden_size) for _ in range(hc_mult)])
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, G, C = x.shape
        normed_chunks = [self.norms[i](x[:, :, i, :]) for i in range(G)]
        x_norm = torch.cat(normed_chunks, dim=-1)
        x_bct = x_norm.transpose(1, 2)
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]
        y_bct = self.act_fn(y_bct)
        y = y_bct.transpose(1, 2).view(B, T, G, C).contiguous()
        return y

class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        shifted_input_ids = input_ids + self.offsets
        return self.embedding(shifted_input_ids)

class Engram(nn.Module):
    def __init__(self, layer_id, hash_mapping_ref):
        super().__init__()
        self.layer_id = layer_id
        self.hash_mapping = hash_mapping_ref
        
        vocab_sizes = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y]
        embed_dim = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram
        
        self.multi_head_embedding = MultiHeadEmbedding(list_of_N=vocab_sizes, D=embed_dim)
        
        self.short_conv = ShortConv(
            hidden_size=backbone_config.hidden_size,
            kernel_size=engram_cfg.kernel_size,
            dilation=engram_cfg.max_ngram_size,
            hc_mult=backbone_config.hc_mult,
        )
        
        engram_hidden_size = (engram_cfg.max_ngram_size-1) * engram_cfg.n_embed_per_ngram
        
        self.value_proj = nn.Linear(engram_hidden_size, backbone_config.hidden_size)
        self.key_projs = nn.ModuleList([
            nn.Linear(engram_hidden_size, backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)
        ])
        
        self.norm1 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])

    def forward(self, hidden_states, input_ids):
        device = hidden_states.device
        # Note: In production, hash calc is CPU pre-fetched. Here we do it on-the-fly.
        input_ids_cpu = input_ids.detach().cpu().numpy()
        hash_indices = self.hash_mapping.hash(input_ids_cpu)[self.layer_id]
        hash_indices = torch.from_numpy(hash_indices).to(device)
        
        embeddings = self.multi_head_embedding(hash_indices).flatten(start_dim=-2)
        
        gates_list = []
        for hc_idx in range(backbone_config.hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states[:,:,hc_idx,:]
            normed_query = self.norm2[hc_idx](query)
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(backbone_config.hidden_size)
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates_list.append(gate)
            
        gates = torch.stack(gates_list, dim=2)
        value = self.value_proj(embeddings).unsqueeze(2)
        output = gates * value
        output = output + self.short_conv(output)
        return output

class MiniBackboneLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mix = nn.Linear(hidden_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        self.norm1 = nn.RMSNorm(hidden_size)
        self.norm2 = nn.RMSNorm(hidden_size)
        
    def forward(self, x):
        res = x
        x = self.norm1(x)
        x = self.mix(x) 
        x = x + res
        res = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + res
        return x

class TransformerBlock(nn.Module):
    def __init__(self, layer_id, hash_mapping_ref):
        super().__init__()
        self.backbone = MiniBackboneLayer(backbone_config.hidden_size)
        self.engram = None
        if layer_id in engram_cfg.layer_ids:
            self.engram = Engram(layer_id=layer_id, hash_mapping_ref=hash_mapping_ref)
    
    def forward(self, input_ids, hidden_states):
        if self.engram is not None:
            mem_out = self.engram(hidden_states=hidden_states, input_ids=input_ids)
            hidden_states = hidden_states + mem_out
        hidden_states = self.backbone(hidden_states)
        return hidden_states

class MiniEngramLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.hash_mapping = NgramHashMapping(engram_cfg)
        self.token_embed = nn.Embedding(backbone_config.vocab_size, backbone_config.hidden_size)
        self.input_proj = nn.Linear(backbone_config.hidden_size, backbone_config.hidden_size * backbone_config.hc_mult)
        self.layers = nn.ModuleList([
            TransformerBlock(layer_id=i, hash_mapping_ref=self.hash_mapping) 
            for i in range(backbone_config.num_layers)
        ])
        self.output_proj = nn.Linear(backbone_config.hidden_size * backbone_config.hc_mult, backbone_config.hidden_size)
        self.lm_head = nn.Linear(backbone_config.hidden_size, backbone_config.vocab_size, bias=False)
        
    def forward(self, input_ids):
        x = self.token_embed(input_ids)
        x = self.input_proj(x) 
        B, L, _ = x.shape
        x = x.view(B, L, backbone_config.hc_mult, backbone_config.hidden_size)
        for layer in self.layers:
            x = layer(input_ids=input_ids, hidden_states=x)
        x = x.flatten(start_dim=2)
        x = self.output_proj(x)
        logits = self.lm_head(x)
        return logits

# ==========================================
# 4. DDP Utilities & Dataset
# ==========================================

# Simple Synthetic Dataset to feed 8 A100s
class SyntheticDataset(Dataset):
    def __init__(self, size=1000, length=128):
        self.size = size
        self.length = length
        # Just random tokens for load testing
        self.data = torch.randint(0, backbone_config.vocab_size, (size, length))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx]

def get_rank():
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def setup_distributed():
    # Initializes the distributed backend which aligns with 'torchrun' environment variables
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    dist.destroy_process_group()

# ==========================================
# 5. DDP Training Loop
# ==========================================
def train_ddp():
    # 1. Setup DDP
    local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    if is_main_process():
        print(f" Initializing DDP on {torch.cuda.device_count()} GPUs")
        print(f" Using NCCL backend on A100s")

    # 2. Model & DDP Wrapper
    model = MiniEngramLLM().to(device)
    # IMPORTANT: wrap model in DDP
    # device_ids ensures this process only uses its assigned GPU
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 3. Data Loading (Must use DistributedSampler)
    # Generate enough data for 8 GPUs
    dataset = SyntheticDataset(size=1024, length=128) 
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=2, pin_memory=True)
    
    # 4. Training Loop
    model.train()
    
    if is_main_process():
        print(" Starting Training Loop...")

    epochs = 2 
    for epoch in range(epochs):
        # Crucial: set epoch for sampler to shuffle data differently each epoch
        sampler.set_epoch(epoch)
        
        for step, input_ids in enumerate(dataloader):
            input_ids = input_ids.to(device, non_blocking=True)
            
            # Create targets (shifted)
            targets = input_ids.clone()
            
            optimizer.zero_grad()
            logits = model(input_ids)
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0 and is_main_process():
                # On A100, we might want to log more frequent, but for demo keep it simple
                print(f"   Epoch {epoch} | Step {step:03d} | Loss: {loss.item():.4f} | GPU Mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    if is_main_process():
        print(" Training Complete!")
        
    # Cleanup
    cleanup_distributed()

if __name__ == '__main__':
    train_ddp()