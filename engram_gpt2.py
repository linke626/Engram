import torch
import torch.nn as nn
import math
import numpy as np
from dataclasses import dataclass, field
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel
# 复用你之前代码中的 Hashing 逻辑，为了简洁，这里假设你把之前的 NgramHashMapping 类保存在了 utils.py
# 如果没有，请把之前代码里的 NgramHashMapping, CompressedTokenizer 等类粘贴到这个文件头部
from engram_train_v1 import NgramHashMapping, CompressedTokenizer 

@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "/data2/home/wanghaoyi/models/gpt2_local"
    engram_vocab_size: List[int] = field(default_factory=lambda: [200000, 200000]) # 加大词表存知识
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 768  # 与 GPT-2 hidden_size 对齐
    n_head_per_ngram: int = 12    # 与 GPT-2 num_heads 对齐
    layer_ids: List[int] = field(default_factory=lambda: [1, 5]) # 在第2层和第6层插入
    pad_id: int = 50256
    seed: int = 42
    kernel_size: int = 4
    hidden_size: int = 768        # GPT-2 Small 默认参数

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
        
        # 投影层：将 n-gram 特征投影回 GPT-2 的隐空间
        input_dim = (engram_cfg.max_ngram_size - 1) * engram_cfg.n_embed_per_ngram
        self.proj = nn.Linear(input_dim, engram_cfg.hidden_size)
        self.gate = nn.Linear(engram_cfg.hidden_size * 2, engram_cfg.hidden_size) # 简单的门控
        
    def forward(self, hidden_states, input_ids):
        # 1. Hash Lookup
        device = hidden_states.device
        # 注意：生产环境应在 CPU 预处理 Hash，这里为了 Demo 在线计算
        input_ids_cpu = input_ids.detach().cpu().numpy()
        hashes = self.hash_mapping.hash(input_ids_cpu)[self.layer_id]
        hashes = torch.from_numpy(hashes).to(device)
        
        # 2. Retrieve
        mem_embeds = self.mh_embed(hashes).flatten(start_dim=-2) # [B, L, D_mem]
        mem_out = self.proj(mem_embeds) # [B, L, D_gpt]
        
        # 3. Simple Gating (融合记忆与当前上下文)
        # 论文中使用了更复杂的 Attention 门控，这里用 Sigmoid 门控模拟
        concat = torch.cat([hidden_states, mem_out], dim=-1)
        g = torch.sigmoid(self.gate(concat))
        
        return g * mem_out

# ================= 包装器：将 Engram 注入 GPT-2 Block =================
class EngramGPT2BlockWrapper(nn.Module):
    def __init__(self, original_block, engram_module):
        super().__init__()
        self.block = original_block
        self.engram = engram_module
        
    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False):
        # 1. 正常执行 GPT-2 Layer
        outputs = self.block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states_out = outputs[0]
        
        # 2. 获取原始 Input IDs (这是一个 Hack，通常需要从外部传入，但为了简便我们假设能访问到 global input)
        # 在训练 Loop 中，我们需要想办法把 input_ids 传进来。
        # 为了不破坏 HF 接口，我们通常在 Model 级别处理，或者通过 global context (不推荐)。
        # **最佳实践**：我们在 Model 级别修改，而不是 Block 级别。
        # 这里为了演示，我们只返回 engram 模块，具体的 forward 在 Model 里写。
        return outputs

# ================= 最终模型：Hybrid Model =================
class HybridEngramGPT2(nn.Module):
    def __init__(self):
        super().__init__()
        print("Loading local GPT-2...")
        self.backbone = AutoModelForCausalLM.from_pretrained(engram_cfg.tokenizer_name_or_path)
        self.hash_mapping = NgramHashMapping(engram_cfg)
        
        # 初始化 Engram 模块字典
        self.engram_layers = nn.ModuleDict()
        for layer_id in engram_cfg.layer_ids:
            self.engram_layers[str(layer_id)] = EngramModule(layer_id, self.hash_mapping)
            
        # 冻结 GPT-2 参数
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        print("GPT-2 parameters frozen. Engram initialized.")

    def forward(self, input_ids, labels=None):
        # 我们手动去跑 GPT-2 的 transformer 层，以便插入 Engram
        # 1. Embedding
        inputs_embeds = self.backbone.transformer.wte(input_ids) + self.backbone.transformer.wpe(torch.arange(input_ids.size(1), device=input_ids.device))
        inputs_embeds = self.backbone.transformer.drop(inputs_embeds)
        
        hidden_states = inputs_embeds
        
        # 2. Iterate Layers
        for i, block in enumerate(self.backbone.transformer.h):
            # GPT-2 Block Forward
            outputs = block(hidden_states)
            hidden_states = outputs[0]
            
            # 3. Engram Injection
            if str(i) in self.engram_layers:
                mem_out = self.engram_layers[str(i)](hidden_states, input_ids)
                hidden_states = hidden_states + mem_out # Residual connection
        
        # 4. Norm & Head
        hidden_states = self.backbone.transformer.ln_f(hidden_states)
        logits = self.backbone.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return (loss, logits) if loss is not None else logits