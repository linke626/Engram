import torch
import torch.nn as nn
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List
from sympy import isprime
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from tokenizers import normalizers, Regex 

# ================= Configuration =================
@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "/data2/home/wanghaoyi/models/Meta-Llama-3-8B-Instruct"
    # Llama-3 vocab is ~128k, Engram vocab slightly larger to avoid collisions
    engram_vocab_size: List[int] = field(default_factory=lambda: [200000, 200000]) 
    max_ngram_size: int = 3
    # N_embed needs to match Llama hidden size (4096) eventually, or use projection.
    # Here we define internal engram embed size.
    n_embed_per_ngram: int = 4096  
    n_head_per_ngram: int = 32     
    # Llama-3-8B has 32 layers. Let's pick a few.
    layer_ids: List[int] = field(default_factory=lambda: [10, 20]) 
    pad_id: int = 128001 # Llama-3 pad_token_id (usually reserved_special_token_0 or eos)
    seed: int = 42
    kernel_size: int = 4
    hidden_size: int = 4096 # Llama-3-8B hidden size

engram_cfg = EngramConfig()

# ================= 移植自 engram_train_v1.py 的组件 =================

class CompressedTokenizer:
    def __init__(self, tokenizer_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])
        
        self.lookup_table, self.num_new_token = self._build_lookup_table()
    
    def __len__(self):
        return self.num_new_token
    
    def _build_lookup_table(self):
        old2new = {}
        key2new = {}          
        new_tokens = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            try:
                text = self.tokenizer.decode([tid], skip_special_tokens=False)
            except:
                text = ""
            
            if "" in text:
                key = str(tid) # Fallback for special tokens
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new.get(tid, 0)

        return lookup, len(new_tokens)
    
    def _compress(self, input_ids):
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        # Safety check for vocab bounds
        valid_ids = np.clip(valid_ids, 0, len(self.lookup_table)-1)
        out[pos_mask] = self.lookup_table[valid_ids]
        return out   
    
    def __call__(self, input_ids):
        return self._compress(input_ids)

def find_next_prime(start, seen_primes):
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1

class NgramHashMapping:
    def __init__(
        self, 
        engram_vocab_size,
        max_ngram_size,
        n_embed_per_ngram,
        n_head_per_ngram,
        layer_ids,
        tokenizer_name_or_path,
        pad_id,
        seed,  
    ):
        self.vocab_size_per_ngram = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_embed_per_ngram = n_embed_per_ngram
        self.n_head_per_ngram = n_head_per_ngram
        self.pad_id = pad_id
        self.layer_ids = layer_ids

        self.compressed_tokenizer = CompressedTokenizer(
            tokenizer_name_or_path=tokenizer_name_or_path
        )            
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)
        if self.pad_id is not None:
            # Check bounds before lookup
            if self.pad_id < len(self.compressed_tokenizer.lookup_table):
                self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])
            else:
                self.pad_id = 0

        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007
        
        self.layer_multipliers = {}

        for layer_id in self.layer_ids:
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,),
                dtype=np.int64
            )
            multipliers = r * 2 + 1
            self.layer_multipliers[layer_id] = multipliers

        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        seen_primes = set()
        vocab_size_across_layers = {}
        
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                num_head = self.n_head_per_ngram
                current_prime_search_start = vocab_size - 1
                
                for _ in range(num_head):
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
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
            shifted = np.pad(x, ((0, 0), (k, 0)),
                                mode='constant', constant_values=self.pad_id)[:, :T]
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
            
            # Vectorized modulo across heads is slightly tricky without broadcasting
            # Loop over heads is safer for correctness
            for j in range(self.n_head_per_ngram):
                mod = int(head_vocab_sizes[j])
                head_hash = mix % mod
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        input_ids = self.compressed_tokenizer(input_ids)
        hash_ids_for_all_layers = {}
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
        return hash_ids_for_all_layers

# ================= Modified Modules for [B, L, D] =================

class ShortConv(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        kernel_size: int = 4, 
        dilation: int = 1, 
        norm_eps: float = 1e-5,
        activation: bool = True,
    ):
        super().__init__()
        self.activation = activation
        
        # Modified: Removed hc_mult logic. Input is [B, L, D]
        # Conv1d expects [B, Channels, Length]
        self.conv = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size, # Depthwise convolution
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        self.norm = nn.RMSNorm(hidden_size, eps=norm_eps)
        
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, L, D)
        Output: (B, L, D)
        """
        B, T, C = x.shape
        
        # Norm
        x_norm = self.norm(x)
        
        # Transpose for Conv1d: [B, C, T]
        x_bct = x_norm.transpose(1, 2)
        
        # Conv
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T] # Causal trimming (assuming left padding logic in conv setup)

        if self.activation:
            y_bct = self.act_fn(y_bct)
            
        # Transpose back: [B, T, C]
        y = y_bct.transpose(1, 2).contiguous()
        
        return y

class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        offsets = [0]
        for n in list_of_N[:-1]: offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.embedding = nn.Embedding(sum(list_of_N), D)
        
    def forward(self, input_ids):
        # input_ids: [B, L, Num_Heads]
        return self.embedding(input_ids + self.offsets)

class EngramModule(nn.Module):
    def __init__(self, layer_id, hash_mapping):
        super().__init__()
        self.layer_id = layer_id
        self.hash_mapping = hash_mapping
        
        # Setup dims
        vocab_sizes = [x for y in hash_mapping.vocab_size_across_layers[layer_id] for x in y]
        embed_dim = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram
        
        self.mh_embed = MultiHeadEmbedding(vocab_sizes, embed_dim)
        
        # Input dim from embedding: (max_ngram - 1) * n_embed
        input_dim = (engram_cfg.max_ngram_size - 1) * engram_cfg.n_embed_per_ngram
        
        # Projection to Model Dimension
        self.proj = nn.Linear(input_dim, engram_cfg.hidden_size)
        
        # ShortConv (Modified for [B, L, D])
        self.short_conv = ShortConv(
            hidden_size=engram_cfg.hidden_size,
            kernel_size=engram_cfg.kernel_size,
            dilation=engram_cfg.max_ngram_size
        )
        
        # Gating mechanism (Adapting from v1 but simplifying for flat D)
        # Original v1 used Key/Query projections. Here we use a simpler Gated update 
        # to mix Engram info into the Backbone stream.
        self.gate_proj = nn.Linear(engram_cfg.hidden_size * 2, engram_cfg.hidden_size) 
        
    def forward(self, hidden_states, input_ids):
        """
        hidden_states: [B, L, D] (Llama hidden states)
        input_ids: [B, L]
        """
        device = hidden_states.device
        
        # 1. Hashing (CPU Bridge)
        if isinstance(input_ids, torch.Tensor):
            input_ids_cpu = input_ids.detach().cpu().numpy()
        else:
            input_ids_cpu = input_ids
            
        hashes = self.hash_mapping.hash(input_ids_cpu)[self.layer_id]
        hashes = torch.from_numpy(hashes).to(device) # [B, L, Num_Heads]
        
        # 2. Retrieve & Project
        # [B, L, Num_Heads, D_head] -> flatten -> [B, L, D_engram]
        mem_embeds = self.mh_embed(hashes).flatten(start_dim=-2) 
        mem_val = self.proj(mem_embeds) # [B, L, D_model]
        
        # 3. ShortConv refinement
        mem_refined = self.short_conv(mem_val)
        
        # 4. Gating / Mixing
        # Mix mem_refined into hidden_states
        concat = torch.cat([hidden_states, mem_refined], dim=-1)
        g = torch.sigmoid(self.gate_proj(concat))
        
        return g * mem_refined

# ... (前面的 imports 和 Config 保持不变) ...
# ... (CompressedTokenizer, NgramHashMapping, ShortConv 等保持不变) ...

# ================= 核心修改：引入 Context 类解决递归问题 =================

class EngramContext:
    """
    一个简单的普通 Python 类，用于在父模型和子层之间共享状态 (input_ids)。
    因为它不是 nn.Module，所以不会导致 PyTorch 的递归死循环。
    """
    def __init__(self):
        self.current_input_ids = None

class EngramLlamaLayerWrapper(nn.Module):
    def __init__(self, original_layer, engram_module, context):
        super().__init__()
        self.original_layer = original_layer
        self.engram_module = engram_module
        # 这里存储的是 EngramContext 实例，它不是 nn.Module，
        # 所以 model.to() 不会递归进这里，打破了循环引用。
        self.context = context 

    def forward(self, hidden_states, *args, **kwargs):
        # 1. 执行原始 Llama 层
        original_outputs = self.original_layer(hidden_states, *args, **kwargs)
        
        if isinstance(original_outputs, tuple):
            hidden_states_out = original_outputs[0]
        else:
            hidden_states_out = original_outputs

        # 2. 从 Context 中获取 input_ids
        input_ids = self.context.current_input_ids
        
        # 如果没有 input_ids (例如非训练调用)，直接返回
        if input_ids is None:
            return original_outputs
            
        # 3. Engram Forward
        engram_out = self.engram_module(hidden_states_out, input_ids)
        
        # 4. 残差连接
        final_hidden = hidden_states_out + engram_out
        
        # 5. 重新打包返回值
        if isinstance(original_outputs, tuple):
            return (final_hidden,) + original_outputs[1:]
        else:
            return final_hidden

class HybridEngramLlama(nn.Module):
    def __init__(self):
        super().__init__()
        print(f"Loading Llama-3 from {engram_cfg.tokenizer_name_or_path}...")
        
        self.backbone = AutoModelForCausalLM.from_pretrained(
            engram_cfg.tokenizer_name_or_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size=engram_cfg.max_ngram_size,
            n_embed_per_ngram=engram_cfg.n_embed_per_ngram,
            n_head_per_ngram=engram_cfg.n_head_per_ngram,
            layer_ids=engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id=engram_cfg.pad_id,
            seed=engram_cfg.seed
        )
        
        # 【关键修改】初始化共享上下文对象
        self.engram_context = EngramContext()

        self.engram_layers = nn.ModuleDict() 
        
        print("Injecting Engram layers...")
        for layer_id in engram_cfg.layer_ids:
            engram_module = EngramModule(layer_id, self.hash_mapping).to(dtype=torch.bfloat16)
            
            original_layer = self.backbone.model.layers[layer_id]
            
            # 【关键修改】传入 self.engram_context 而不是 self
            wrapped_layer = EngramLlamaLayerWrapper(original_layer, engram_module, self.engram_context)
            
            self.backbone.model.layers[layer_id] = wrapped_layer
            
            # 这里的引用仅用于管理参数，不会造成结构性递归问题
            self.engram_layers[str(layer_id)] = engram_module

        # 冻结骨干网络参数
        for name, param in self.backbone.named_parameters():
            if "engram_module" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            
        print("Llama-3 parameters frozen. Engram initialized and injected.")

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        # 1. 将 input_ids 更新到共享的 context 对象中
        self.engram_context.current_input_ids = input_ids
        
        # 2. 调用 backbone
        try:
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=False,
                return_dict=True,
                **kwargs
            )
        finally:
            # 3. 清理 context，防止引用残留（可选，但推荐）
            self.engram_context.current_input_ids = None
        
        if labels is not None:
            return outputs.loss, outputs.logits
        return outputs.logits