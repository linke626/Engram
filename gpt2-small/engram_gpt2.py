import torch
import torch.nn as nn
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
# 复用你之前代码中的 Hashing 逻辑
from engram_train_v1 import NgramHashMapping, CompressedTokenizer 

@dataclass
class EngramConfig:
    # 确保这里的路径是你本地真实存在的，或者改成 "gpt2"
    tokenizer_name_or_path: str = "/data2/home/wanghaoyi/models/gpt2_local"
    engram_vocab_size: List[int] = field(default_factory=lambda: [200000, 200000]) 
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 768  
    n_head_per_ngram: int = 12    
    layer_ids: List[int] = field(default_factory=lambda: [1, 5]) 
    pad_id: int = 50256
    seed: int = 42
    kernel_size: int = 4
    hidden_size: int = 768        

engram_cfg = EngramConfig()

# ================= Engram 模块定义 (适配 GPT-2) =================
class MultiHeadEmbedding(nn.Module):
    def __init__(self, list_of_N, D):
        super().__init__()
        offsets = [0]
        for n in list_of_N[:-1]: offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.embedding = nn.Embedding(sum(list_of_N), D)
    def forward(self, input_ids):
        return self.embedding(input_ids + self.offsets)

class EngramModule(nn.Module):
    def __init__(self, layer_id, hash_mapping):
        super().__init__()
        self.layer_id = layer_id
        self.hash_mapping = hash_mapping
        
        # 动态获取哈希表大小
        vocab_sizes = [x for y in hash_mapping.vocab_size_across_layers[layer_id] for x in y]
        embed_dim = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram
        
        self.mh_embed = MultiHeadEmbedding(vocab_sizes, embed_dim)
        
        # 投影层
        input_dim = (engram_cfg.max_ngram_size - 1) * engram_cfg.n_embed_per_ngram
        self.proj = nn.Linear(input_dim, engram_cfg.hidden_size)
        self.gate = nn.Linear(engram_cfg.hidden_size * 2, engram_cfg.hidden_size) 
        
    def forward(self, hidden_states, input_ids):
        # 1. Hash Lookup
        device = hidden_states.device
        input_ids_cpu = input_ids.detach().cpu().numpy()
        hashes = self.hash_mapping.hash(input_ids_cpu)[self.layer_id]
        hashes = torch.from_numpy(hashes).to(device)
        
        # 2. Retrieve
        mem_embeds = self.mh_embed(hashes).flatten(start_dim=-2) # [B, L, D_mem]
        mem_out = self.proj(mem_embeds) # [B, L, D_gpt]
        
        # 3. Simple Gating
        concat = torch.cat([hidden_states, mem_out], dim=-1)
        g = torch.sigmoid(self.gate(concat))
        
        return g * mem_out

# ================= 最终模型：Hybrid Model =================
class HybridEngramGPT2(nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading local GPT-2...")
        self.backbone = AutoModelForCausalLM.from_pretrained(engram_cfg.tokenizer_name_or_path)
        
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
        
        # 初始化 Engram 模块字典
        self.engram_layers = nn.ModuleDict()
        for layer_id in engram_cfg.layer_ids:
            self.engram_layers[str(layer_id)] = EngramModule(layer_id, self.hash_mapping)
            
        # 冻结 GPT-2 参数
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        print("GPT-2 parameters frozen. Engram initialized.")

    def forward(self, input_ids, labels=None):
        inputs_embeds = self.backbone.transformer.wte(input_ids) + self.backbone.transformer.wpe(torch.arange(input_ids.size(1), device=input_ids.device))
        inputs_embeds = self.backbone.transformer.drop(inputs_embeds)
        
        hidden_states = inputs_embeds
        
        for i, block in enumerate(self.backbone.transformer.h):
            outputs = block(hidden_states)
            hidden_states = outputs[0]
            
            if str(i) in self.engram_layers:
                mem_out = self.engram_layers[str(i)](hidden_states, input_ids)
                hidden_states = hidden_states + mem_out 
        
        hidden_states = self.backbone.transformer.ln_f(hidden_states)
        logits = self.backbone.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return (loss, logits) if loss is not None else logits