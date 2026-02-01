"""
================================================================================
[Engram Architecture Demo Implementation]

DISCLAIMER:
1. Demo Purpose Only: 
   This code is a demonstration version intended to illustrate the core logic and 
   data flow of the Engram module.

2. Production Readiness: 
   This implementation requires further optimization for actual production use 
   (e.g., custom CUDA kernels, distributed training support).

3. Simplifications: 
   Standard components (Normalization, Attention, MoE) and complex Hyper-connection 
   mechanisms are omitted or mocked in this version to focus exclusively on the 
   Engram module implementation.
================================================================================
"""

"""
pip install torch numpy transformers sympy
"""

## built-in
from typing import List
from dataclasses import dataclass, field
import math

## third-party
from sympy import isprime
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex 

@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280*5, 129280*5])
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4
    
@dataclass
class BackBoneConfig:
    hidden_size: int = 1024
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30
    
engram_cfg = EngramConfig()
backbone_config = BackBoneConfig()

class CompressedTokenizer:
    def __init__(
        self,
        tokenizer_name_or_path,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        
        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),             # 1. Unicode 规范化 (把各种奇怪的字符变标准，如全角变半角)
            normalizers.NFD(),              # 2. 字符分解 (为去重音做准备)
            normalizers.StripAccents(),     # 3. 去除重音符号 (如 "é" -> "e")
            normalizers.Lowercase(),        # 4. 转小写 ("The" -> "the")
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "), # 5. 把连续空白符合并成一个空格           
            # 我们想用 Strip() 去除单词前后的空格（比如 " apple " -> "apple"）。
            # 但是，有的 Token 本身就是一个空格（ID: 220）。如果直接 Strip()，这个 Token 就变成空字符串 "" 消失了！
            # 解决：先检测：如果它是纯空格，先把它变成一个替身（SENTINEL: \uE000）。再执行 Strip()：此时替身不会被删掉。最后还原：把替身变回空格。 
            # --- 哨兵机制 (Sentinel) ---
            normalizers.Replace(Regex(r"^ $"), SENTINEL), # 6. 如果 Token 本身就是个空格，先把它变成特殊符号
            normalizers.Strip(),                          # 7. 去除首尾空格
            normalizers.Replace(SENTINEL, " "),           # 8. 把特殊符号变回空格
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
            text = self.tokenizer.decode([tid], skip_special_tokens=False)
            
            if "�" in text:
                # 有些 Token 不是合法的 UTF-8 (比如 BPE 的字节片段)，这种保留原样
                # 如果经过归一化后，这个词竟然消失了（变成空字符串），那为了防止出Bug，我们用回它原始的样子
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text
                
            nid = key2new.get(key)
            if nid is None:
                # 如果这个字符串第一次见，给它发个新 ID
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid
        
        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)
    
    def _compress(self, input_ids):
        # 利用 Numpy 的数组索引功能，实现 O(1) 的并行查表
        # lookup_table[ [10, 11, 13] ]  ==>  [0, 0, 1]
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        valid_ids = arr[pos_mask]
        out[pos_mask] = self.lookup_table[valid_ids]
        return out   
    
    def __call__(self, input_ids):
        return self._compress(input_ids)
            
class ShortConv(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        kernel_size: int = 4, 
        dilation: int = 1, 
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation
        
        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            # 这是一个 深度卷积。
            # 普通卷积: 输入的每个通道都会和所有卷积核做运算，参数量巨大。
            # 深度卷积: 每个通道只被自己的卷积核处理，通道之间不发生交互。
            # 目的: 极大地减少参数量和计算量。这里的卷积只是为了在时间维度上混合信息，不需要在通道维度上混合（通道混合交给后面的 Linear 层去做）。
            groups=total_channels,
            bias=False,
            # padding=(kernel_size - 1) * dilation (因果填充)
            # 这是为了实现 “因果性” (Causality)。目标: 第 t 个时刻的输出，只能依赖 t 及 t 之前的输入，绝不能偷看 t+1 之后的未来信息。
            # 做法: 在序列的最左边（开头）填充足够多的 0。如果 kernel_size=4，它会在左边填 3 个 0。这样卷积核滑到第 1 个词时，看到的是 [0, 0, 0, Word1]，刚好填满窗口，不会越界。
            # 注意: PyTorch 的 padding 参数实际上是双边填充（左右都填）。所以代码在 forward 里必须要有一个 [:T] 的切片操作，把右边多填出来的“未来信息”切掉。
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )

        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps) 
            for _ in range(hc_mult)
        ])
        
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B,L,HC_MULT,D)
        Output: (B,L,HC_MULT,D)
        """
        B, T, G, C = x.shape
        
        assert G == self.hc_mult, f"Input groups {G} != hc_mult {self.hc_mult}"

        normed_chunks = []
        # 输入 x: [B, T, G, C]
        for i in range(G):
            chunk = x[:, :, i, :]
            normed_chunks.append(self.norms[i](chunk))
        x_norm = torch.cat(normed_chunks, dim=-1) 
        # 输出 x_norm: [B, T, G*C] (把 G 和 C 拍扁在一起)
        
        x_norm = torch.cat(normed_chunks, dim=-1)
        x_bct = x_norm.transpose(1, 2) 
        # 变换为 [B, G*C, T]，因为 Conv1d 要求时间维在最后
        y_bct = self.conv(x_bct)      # 输出长度变成了 T + padding
        """
        1. 原始输入 (T=3):
        [A, B, C]

        2. PyTorch 双边填充后:
        [0, 0, A, B, C, 0, 0]
         ^^^^  <-- 左边补的 (这是我们想要的，代表“过去没有信息”)
                        ^^^^ <-- 右边补的 (这是多余的，代表“未来”)

        3. 卷积后的输出 (假设长度变为了 3 + 2 = 5):
        [Out1, Out2, Out3, Out4, Out5]
            Let's say:
            Out1 对应 [0,0,A] 的卷积结果
            Out2 对应 [0,A,B] 的卷积结果
            Out3 对应 [A,B,C] 的卷积结果  <-- 到这里我们就该停了！(对应原长度 T=3)
            -----------------------------
            Out4 对应 [B,C,0] 的卷积结果  <-- 这是由右边填充产生的，切掉！
            Out5 对应 [C,0,0] 的卷积结果  <-- 切掉！

        4. 执行 y_bct[..., :3] 切片后:
        [Out1, Out2, Out3] 
        """
        y_bct = y_bct[..., :T]        # 关键：切掉尾巴，只保留前 T 个

        if self.activation:
            y_bct = self.act_fn(y_bct)    # SiLU 激活
        y = y_bct.transpose(1, 2).view(B, T, G, C) # 变回 [B, T, G, C]
        
        return y

# 找到一个比给定数字大的、且没有被使用过的最小素数
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
            self.pad_id = int(self.compressed_tokenizer.lookup_table[self.pad_id])

        # --- 防止溢出的安全界限 ---
        max_long = np.iinfo(np.int64).max
        # M_max 算的是：一个乘数最大能是多少，才能保证 (Token_ID * Multiplier) 不会溢出 64位整数？
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)

        PRIME_1 = 10007 # 一个用来打散种子的素数

        self.layer_multipliers = {}

        # 为每一层生成一套独特的乘数
        for layer_id in self.layer_ids:
            # 1. 混合种子：保证每一层用的随机数都不一样
            base_seed = int(seed + PRIME_1 * int(layer_id))
            g = np.random.default_rng(base_seed)
            
            # 2. 生成随机数
            r = g.integers(
                low=0,
                high=half_bound,
                size=(self.max_ngram_size,), # 生成 3 个随机数 (对应 3-gram)
                dtype=np.int64
            )
            
            # 3. 强制变成奇数 (Multiplier = r * 2 + 1)
            # 数学原理：在计算机里，奇数乘法是“可逆”的，能保留更多的信息熵。
            # 如果乘数是偶数，乘多了低位会全是0，导致信息丢失。
            multipliers = r * 2 + 1
        
            self.layer_multipliers[layer_id] = multipliers
        
        # 确定“每个哈希头具体要多大的表”。比如 Head 1 用 100003，Head 2 用 100019 等。
        self.vocab_size_across_layers = self.calculate_vocab_size_across_layers()

    def calculate_vocab_size_across_layers(self):
        """
        计算每一层、每种 N-gram、每个哈希头所需的词表大小（即哈希取模的基数）。
        
        核心逻辑：
        为了最大化哈希分布的随机性并减少冲突，我们需要确保不同的 Layer 和不同的 Head 
        使用不同的素数作为哈希表的大小。
        
        例如：
        - Layer 1 的 Head 0 可能使用 100003
        - Layer 1 的 Head 1 可能使用 100019
        - Layer 15 的 Head 0 可能使用 100043
        这样即使两个词组算出了一样的哈希值，在取模后也会被分散到不同的位置。
        """
        
        # 用于记录全局已经使用过的素数，防止不同层或不同头复用同一个素数导致哈希相关性过高
        seen_primes = set()
        
        # 最终结果字典，结构为：{layer_id: [ [2-gram的各头大小], [3-gram的各头大小] ]}
        vocab_size_across_layers = {}
        
        # 1. 遍历每一层 (例如 Layer 1, Layer 15)
        for layer_id in self.layer_ids:
            all_ngram_vocab_sizes = []
            
            # 2. 遍历每种 N-gram 长度 (从 2-gram 到 max_ngram_size)
            # 注意：range(2, 4) 会产生 [2, 3]
            for ngram in range(2, self.max_ngram_size + 1):
                current_ngram_heads_sizes = []
                
                # 获取配置中设定的基准大小 (例如 129280*5)
                # ngram - 2 是因为列表从 0 开始存 2-gram 的配置
                vocab_size = self.vocab_size_per_ngram[ngram - 2]
                
                # 获取哈希头的数量 (例如 8)
                num_head = self.n_head_per_ngram
                
                # 设定搜索起跑线：我们希望找到的素数略大于基准大小
                current_prime_search_start = vocab_size - 1
                
                # 3. 为当前的 N-gram 分配 num_head 个不同的素数
                for _ in range(num_head):
                    # 调用辅助函数查找下一个可用的素数
                    # 要求：比 current_prime_search_start 大，且不在 seen_primes 中
                    found_prime = find_next_prime(
                        current_prime_search_start, 
                        seen_primes
                    )
                    
                    # 记录这个素数已使用
                    seen_primes.add(found_prime)
                    
                    # 将其加入当前 N-gram 的配置列表
                    current_ngram_heads_sizes.append(found_prime)
                    
                    # 更新起跑线，确保下一个头找到的素数比当前这个更大
                    current_prime_search_start = found_prime
                
                # 将当前 N-gram 的所有头大小列表加入总列表
                all_ngram_vocab_sizes.append(current_ngram_heads_sizes)
            
            # 记录该层的所有配置
            vocab_size_across_layers[layer_id] = all_ngram_vocab_sizes
            
        return vocab_size_across_layers

    def _get_ngram_hashes(
        self,
        input_ids: np.ndarray,
        layer_id: int,
    ) -> np.ndarray:
        """
        核心哈希计算函数（向量化实现）。
        
        功能：
        将输入的 token 序列（如 "A B C D"）转换为 N-gram 哈希索引。
        它会并行计算 2-gram (如 AB, BC, CD) 和 3-gram (如 ABC, BCD) 的哈希值。
        
        参数:
        - input_ids: 压缩后的 token ID 矩阵，形状 [Batch, Time]
        Time 指的是 Sequence Length（序列长度）
        - layer_id: 当前处于模型的哪一层（不同层使用不同的哈希参数）
        
        返回:
        - stack: 哈希索引矩阵，形状 [Batch, Time, Total_Heads]
          其中 Total_Heads = (种 N-gram) * (每种 N-gram 的头数)
        """
        
        # 1. 准备数据
        # 确保输入是 int64 类型，防止哈希计算中溢出
        x = np.asarray(input_ids, dtype=np.int64)
        B, T = x.shape

        # 获取当前层专属的随机乘数 (Multipliers)，用于哈希计算
        # 形状通常为 [max_ngram_size]，例如 [m1, m2, m3]
        multipliers = self.layer_multipliers[layer_id]

        # 2. 构建错位视图 (Shifted Views)
        # 这一步通过“平移”操作，快速构建出历史时刻的视图。
        # 
        # 例如输入序列: [A, B, C, D] (即 k=0)
        # k=1 的平移: [Pad, A, B, C] (每个位置的前 1 个词)
        # k=2 的平移: [Pad, Pad, A, B] (每个位置的前 2 个词)
        def shift_k(k: int) -> np.ndarray:
            if k == 0: return x
            # np.pad 用于在左侧填充 k 个 Pad ID (值为 self.pad_id)
            # [:, :T] 用于切掉右侧多出来的部分，保持长度仍为 T
            shifted = np.pad(x, ((0, 0), (k, 0)),
                                mode='constant', constant_values=self.pad_id)[:, :T]
            return shifted

        # 一次性生成所有需要的历史视图
        # base_shifts[0] 是当前词，base_shifts[1] 是前一个词，以此类推
        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes = []
        
        # 3. 循环计算每种 N-gram 的哈希值
        # 从 2-gram 开始循环 (因为 1-gram 通常由主模型处理，Engram 负责组合特征)
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            
            # 取出计算 n-gram 所需的前 n 个视图
            # 例如 n=3 时，tokens 包含 [当前词, 前1词, 前2词]
            tokens = base_shifts[:n]
            
            # --- 核心哈希公式 (Polynomial Rolling Hash) ---
            # Hash = (T0 * m0) XOR (T1 * m1) XOR (T2 * m2) ...
            # 这种算法极其适合 GPU/CPU 并行，因为它没有顺序依赖
            
            # 第一步：初始化 mix 为 (当前词 * 乘数0)
            mix = (tokens[0] * multipliers[0])
            
            # 第二步：累积异或历史词
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            
            # --- 多头取模映射 (Multi-head Modulo) ---
            # 同样的一个 mix 值，要被映射到多个不同的哈希表里
            num_heads_for_this_ngram = self.n_head_per_ngram
            
            # 获取当前层、当前 N-gram 对应的所有素数模数列表
            head_vocab_sizes = self.vocab_size_across_layers[layer_id][n_gram_index]
            
            for j in range(num_heads_for_this_ngram):
                # 取出第 j 个头的素数大小
                mod = int(head_vocab_sizes[j])
                
                # 取模运算，得到最终的内存地址索引
                # 结果范围在 [0, mod-1] 之间
                head_hash = mix % mod
                
                # 存入结果列表，强制转为 int64 以节省内存并保持一致
                all_hashes.append(head_hash.astype(np.int64, copy=False))
        
        # 4. 堆叠结果
        # 将所有 N-gram 的所有 Head 的哈希值沿最后一个维度堆叠
        # 最终形状: [Batch, Time, Heads]
        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids):
        """
        对外的高层接口。
        
        流程：
        1. 压缩：将原始 Token ID 转换为压缩 ID (归一化、去重音等)。
        2. 路由：为配置好的每一层 (layer_ids) 计算其专属的哈希索引。
        
        参数:
        - input_ids: 原始模型的 Token IDs
        
        返回:
        - hash_ids_for_all_layers: 字典 {layer_id: np.ndarray([Batch, Time, Heads])}
        """
        # 1. 预处理：Token 压缩
        # 作用：让 "The" 和 "the" 映射到同一个 ID，减少哈希空间的稀疏性
        input_ids = self.compressed_tokenizer(input_ids)
        
        hash_ids_for_all_layers = {}
        
        # 2. 逐层计算
        # 不同层使用不同的 multipliers 和 prime modulus，所以要分开算
        for layer_id in self.layer_ids:
            hash_ids_for_all_layers[layer_id] = self._get_ngram_hashes(input_ids, layer_id=layer_id)
            
        return hash_ids_for_all_layers

class MultiHeadEmbedding(nn.Module):
    """
    多头嵌入层 (Multi-Head Embedding)
    
    功能：
    将多个不同大小的哈希表（对应不同的 N-gram 头）合并到一个巨大的 nn.Embedding 中进行管理。
    这是一种通过通过“地址偏移”来实现并行查表的技巧。
    """
    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D
        
        # --- 计算偏移量 (Offsets) ---
        # 逻辑：
        # 假设 Head 0 大小为 100，Head 1 大小为 200。
        # 那么 Head 0 的索引范围是 [0, 99]。
        # Head 1 的索引范围原本是 [0, 199]，但在合并的大表中，我们要把它平移到 [100, 299]。
        # offsets 数组就是 [0, 100, 300, ...]
        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        
        # register_buffer 用于保存状态（会随模型保存/加载），但不是可训练参数（不会有梯度）
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        
        # 创建一个总的 Embedding 表，包含所有头的容量
        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        参数:
        input_ids: [Batch, Length, Num_Heads] 
                   这是 NgramHashMapping 计算出的原始哈希索引（每个头都从 0 开始）。
        """
        # --- 加上偏移量 ---
        # 利用 PyTorch 的广播机制 (Broadcasting)：
        # input_ids [..., Num_Heads] + offsets [Num_Heads]
        # 这样每个头的索引就被“平移”到了大表中对应的段落。
        shifted_input_ids = input_ids + self.offsets
        
        # 查表，返回 [Batch, Length, Num_Heads, Embedding_Dim]
        output = self.embedding(shifted_input_ids)
        
        return output
    
class Engram(nn.Module):
    """
    Engram 记忆模块核心类
    
    功能：
    集成哈希计算、记忆检索、门控融合（Gating）和时序平滑（ShortConv）。
    """
    def __init__(self, layer_id):
        super().__init__()
        self.layer_id = layer_id
        
        # --- 1. 初始化哈希映射器 ---
        # 这是一个计算密集型组件，负责把 Token 序列转换成哈希索引
        self.hash_mapping = NgramHashMapping(
            engram_vocab_size=engram_cfg.engram_vocab_size,
            max_ngram_size = engram_cfg.max_ngram_size,
            n_embed_per_ngram = engram_cfg.n_embed_per_ngram,
            n_head_per_ngram = engram_cfg.n_head_per_ngram,
            layer_ids = engram_cfg.layer_ids,
            tokenizer_name_or_path=engram_cfg.tokenizer_name_or_path,
            pad_id = engram_cfg.pad_id,
            seed = engram_cfg.seed,
        )
        
        # --- 2. 初始化嵌入层 ---
        # 获取当前层所有 N-gram 类型、所有头的表大小列表
        # 列表推导式将嵌套列表展平：[[2gram_heads], [3gram_heads]] -> [h1, h2, h3...]
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N = [x for y in self.hash_mapping.vocab_size_across_layers[self.layer_id] for x in y],
            # 向量维度 = 总维度 / 头数 (例如 512 / 8 = 64)
            D = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram,
        )
        
        # --- 3. 初始化短卷积 ---
        # 用于在时间维度上平滑检索到的记忆
        self.short_conv = ShortConv(
            hidden_size = backbone_config.hidden_size,
            kernel_size = engram_cfg.kernel_size,
            dilation    = engram_cfg.max_ngram_size,
            hc_mult     = backbone_config.hc_mult,
        )
        
        # --- 4. 初始化投影层和归一化层 ---
        # 2-gram 和 3-gram 的结果被拼接在一起，所以总维度是 (max_ngram - 1) * n_embed
        # 例如 (3-1) * 512 = 1024
        engram_hidden_size = (engram_cfg.max_ngram_size-1) * engram_cfg.n_embed_per_ngram
        
        # Value 投影：把记忆向量映射回主模型的维度
        self.value_proj = nn.Linear(engram_hidden_size,backbone_config.hidden_size)
        
        # Key 投影 & Norm：用于计算门控 (Gate)
        # DeepSeek 使用了 Multi-Head Latent Attention (MLA) 或类似的超连接结构 (hc_mult)
        # 所以这里使用了 ModuleList 为每个超连接通道单独初始化层
        self.key_projs = nn.ModuleList(
            [nn.Linear(engram_hidden_size,backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)]
        )
        self.norm1 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
        self.norm2 = nn.ModuleList([nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)])
    
    def forward(self,hidden_states,input_ids):
        """
        前向传播
        
        参数:
        hidden_states: [Batch, L, HC_MULT, D] - 主模型的当前隐藏状态（作为 Query）
        input_ids: [Batch, L] - 原始输入的 Token ID
        """
        # 1. 计算哈希索引 (CPU/Numpy 操作)
        # 结果转为 Tensor: [Batch, L, Total_Heads]
        hash_input_ids = torch.from_numpy(self.hash_mapping.hash(input_ids)[self.layer_id])
        # 将 Tensor 移到和模型相同的设备上 (GPU)
        hash_input_ids = hash_input_ids.to(hidden_states.device) 
        
        # 2. 检索记忆向量
        # 输出: [Batch, L, Total_Heads, Head_Dim]
        # flatten(start_dim=-2): 将最后两个维度合并 -> [Batch, L, Total_Heads * Head_Dim]
        # 也就是把所有 N-gram 的记忆拼成一个长向量
        embeddings = self.multi_head_embedding(hash_input_ids).flatten(start_dim=-2)
        
        # 3. 计算门控 (Gating) - 逐通道处理
        gates = []
        for hc_idx in range(backbone_config.hc_mult):
            # --- 计算 Key (记忆的特征) ---
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            
            # --- 获取 Query (当前上下文特征) ---
            query = hidden_states[:,:,hc_idx,:]
            normed_query = self.norm2[hc_idx](query)
            
            # --- 点积计算相似度 ---
            # 类似于 Attention 机制：看当前上下文和检索到的记忆有多匹配
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(backbone_config.hidden_size)
            
            # --- 激进的门控激活 ---
            # abs().clamp_min().sqrt() * sign(): 这是一种保持符号的非线性缩放，增强数值稳定性
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            # Sigmoid: 将值压缩到 [0, 1] 之间，作为“通过率”
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
            
        # 堆叠门控结果: [Batch, L, HC_MULT, 1]
        gates = torch.stack(gates,dim=2)
        
        # 4. 融合记忆
        # Value 投影: 记忆向量 -> 主模型维度
        # 门控机制: 只有 Gate 值高（匹配度高）的记忆才会被保留
        value = gates * self.value_proj(embeddings).unsqueeze(2)
        
        # 5. 时序平滑 & 残差连接
        # output = 原值 + 卷积后的值 (这里实现了一个局部的残差块)
        output = value + self.short_conv(value)
        
        return output 

class TransformerBlock(nn.Module):
    """
    模拟的 Transformer 块
    
    功能：
    演示如何将 Engram 模块“无缝插入”到现有的 Transformer 层中。
    """
    def __init__(self,layer_id):
        super().__init__()
        # 模拟原始的 Attention 和 MLP/MoE 层 (这里用恒等映射 lambda x:x 代替)
        self.attn = lambda x:x
        self.moe  = lambda x:x
        
        self.engram = None
        # --- 条件插入 ---
        # 只有当当前层 ID 在配置列表 (layer_ids) 中时，才初始化 Engram
        # 这体现了 Engram 的稀疏挂载特性（只挂在浅层和中层）
        if layer_id in engram_cfg.layer_ids:
            self.engram = Engram(layer_id=layer_id)
    
    def forward(self,input_ids,hidden_states):
        # --- Engram 介入 ---
        # 如果这一层有 Engram，就先执行 Engram 增强
        if self.engram is not None:
            # 残差连接 (Residual Connection):
            # 新状态 = 旧状态 + Engram检索到的记忆
            hidden_states = self.engram(hidden_states=hidden_states,input_ids=input_ids) + hidden_states
        
        # --- 标准 Transformer 流程 ---
        hidden_states = self.attn(hidden_states) + hidden_states
        hidden_states = self.moe(hidden_states) + hidden_states
        
        return hidden_states

if __name__ == '__main__':
    # --- 1. 构建模拟的大语言模型 (Mock LLM) ---
    # 这里用一个简单的 Python 列表来代表整个模型的层级结构
    LLM = [
        # 第 0 层: 词嵌入层 (Embedding)
        # 将 Token ID 转换为向量。输入 [Batch, Length] -> 输出 [Batch, Length, Hidden_Size]
        nn.Embedding(backbone_config.vocab_size, backbone_config.hidden_size),
        
        # 第 1 到 N 层: Transformer 模块
        # 使用列表推导式创建 num_layers 个 Block。
        # 注意：TransformerBlock 内部会根据 layer_id 判断是否要插入 Engram 模块。
        *[TransformerBlock(layer_id=layer_id) for layer_id in range(backbone_config.num_layers)],
        
        # 最后一层: 语言模型头 (LM Head)
        # 将隐藏状态映射回词表大小，用于预测下一个词。
        # 输入 [Batch, Length, Hidden_Size] -> 输出 [Batch, Length, Vocab_Size]
        nn.Linear(backbone_config.hidden_size, backbone_config.vocab_size)
    ]

    # --- 2. 准备测试数据 ---
    # 这是一句包含专有名词 ("Alexander the Great", "Bucephalus") 的句子，
    # 专门用来测试 Engram 是否能通过 N-gram 记住这些固定搭配。
    text = "Only Alexander the Great could tame the horse Bucephalus."
    
    # 初始化分词器，trust_remote_code=True 允许加载 DeepSeek 的自定义代码
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path, trust_remote_code=True)
    
    # 分词并转换为 PyTorch 张量
    # input_ids 形状: [1, Sequence_Length] (因为只有 1 句话)
    input_ids = tokenizer(text, return_tensors='pt').input_ids

    # 获取 Batch Size (B) 和 序列长度 (L)
    B, L = input_ids.shape

    # --- 3. 模拟前向传播 (Forward Pass) ---
    # 我们手动遍历每一层，模拟数据在模型中的流动过程
    for idx, layer in enumerate(LLM):
        
        # [情况 A]: 处理第一层 (Embedding)
        if idx == 0:
            # 输入 Token ID，得到基础向量。形状: [B, L, D]
            hidden_states = LLM[0](input_ids)
            
            # --- 模拟超连接 (Mock Hyper-connection) ---
            # DeepSeek-V3 采用了 Multi-Head Latent Attention (MLA) 或类似的并行结构。
            # 为了让 Engram 模块能处理这种结构，我们需要将数据从 3D 扩展到 4D。
            # 1. unsqueeze(2): 在第2维插入一个新维度 -> [B, L, 1, D]
            # 2. expand: 复制数据填满这个维度 -> [B, L, hc_mult, D]
            # 现在，数据的形状变成了 [B, L, 4, 1024]，模拟有 4 条并行的信息流。
            hidden_states = hidden_states.unsqueeze(2).expand(-1, -1, backbone_config.hc_mult, -1)      
        
        # [情况 B]: 处理最后一层 (LM Head)
        elif idx == len(LLM)-1:
            # --- 模拟超连接的合并 ---
            # 标准的 Linear 层只接受 2D 或 3D 输入，不能处理 4D 的超连接数据。
            # 这里简单粗暴地只取第 0 个通道的数据 (Mock)，还原回 [B, L, D]
            hidden_states = hidden_states[:,:,0,:] 
            
            # 映射回词表，得到 Logits。形状: [B, L, Vocab_Size]
            output = layer(hidden_states)
        
        # [情况 C]: 处理中间的 Transformer 层
        else:
            # 调用 TransformerBlock。
            # 注意：必须同时传入 input_ids，因为 Engram 模块需要用它来计算 N-gram 哈希！
            # hidden_states 形状保持不变: [B, L, hc_mult, D]
            hidden_states = layer(input_ids=input_ids, hidden_states=hidden_states)

    # --- 4. 输出结果验证 ---
    print("✅ Forward Complete!")
    # 打印形状以验证流水线没有断裂，维度变化符合预期
    # 预期 output.shape: torch.Size([1, L, 129280])
    print(f"{input_ids.shape=}\n{output.shape=}")
            