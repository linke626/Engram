import torch
import math
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

# 获取当前脚本文件的上一级目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# 将上一级目录添加到系统搜索路径中
sys.path.append(parent_dir)

from engram_Llama3_8B_Instruct import HybridEngramLlama, engram_cfg

# ================= 配置 =================

# 【修改 1】定义本地模型路径
BASE_MODEL_PATH = "/data2/home/wanghaoyi/models/Meta-Llama-3-8B-Instruct"

# 测试集路径
TEST_FILE = "/data2/home/wanghaoyi/data/wikitext/wiki.test.raw" 
# 训练好的权重文件
CHECKPOINT_PATH = "../engram_llama3_weights.pt" 
# 滑动窗口大小
STRIDE = 512 
# 上下文长度
MAX_LENGTH = 1024 
# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_data(tokenizer, file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 简单的 tokenization
    encodings = tokenizer(text, return_tensors="pt")
    return encodings

def compute_ppl(model, encodings, stride=STRIDE, max_length=MAX_LENGTH):
    """
    计算困惑度 (Perplexity) 的标准方法（Sliding Window）
    """
    model.eval()
    nlls = []
    
    seq_len = encodings.input_ids.size(1)
    prev_end_loc = 0
    
    pbar = tqdm(range(0, seq_len, stride), desc="Evaluating PPL")
    
    for begin_loc in pbar:
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # mask context

        if input_ids.size(1) < 2:
            continue

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            
            if isinstance(outputs, tuple):
                loss = outputs[0]
            else:
                loss = outputs.loss

            nlls.append(loss * trg_len)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    total_tokens = end_loc
    ppl = torch.exp(total_nll / total_tokens)
    return ppl.item()

def main():
    print(f"Using Device: {DEVICE}")
    
    # 【修改 2】覆盖配置中的路径，确保 HybridEngramLlama 也加载本地模型
    print(f"Overwriting engram_cfg path to: {BASE_MODEL_PATH}")
    engram_cfg.tokenizer_name_or_path = BASE_MODEL_PATH

    print(f"Loading Tokenizer from: {BASE_MODEL_PATH}")
    # 【修改 3】使用本地路径加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    
    print(f"Loading Test Data from: {TEST_FILE}")
    encodings = load_data(tokenizer, TEST_FILE)
    print(f"Total tokens in test set: {encodings.input_ids.size(1)}")

    # ---------------------------------------------------------
    # 1. 评估原始 Llama-3 (Baseline)
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("Step 1: Evaluating Original Llama-3 (Baseline)...")
    print("="*40)
    
    # 【修改 4】使用本地路径加载 Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH, 
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(DEVICE)
    
    baseline_ppl = compute_ppl(base_model, encodings)
    print(f"Result -> Original Llama-3 PPL: {baseline_ppl:.4f}")
    
    del base_model
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 2. 评估训练后的 Hybrid Engram 模型
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("Step 2: Evaluating Trained Hybrid Engram Model...")
    print("="*40)
    
    # 初始化模型 (此时 engram_cfg.tokenizer_name_or_path 已被修改为本地路径)
    hybrid_model = HybridEngramLlama().to(DEVICE, dtype=torch.bfloat16)
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights from {CHECKPOINT_PATH}...")
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        keys = hybrid_model.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded. Missing keys: {len(keys.missing_keys)}")
    else:
        print(f"Warning: Checkpoint {CHECKPOINT_PATH} not found! Running with random Engram weights.")

    trained_ppl = compute_ppl(hybrid_model, encodings)
    print(f"Result -> Trained Hybrid Engram PPL: {trained_ppl:.4f}")

    # ---------------------------------------------------------
    # 3. 总结
    # ---------------------------------------------------------
    print("\n" + "="*40)
    print("FINAL COMPARISON")
    print("="*40)
    print(f"Original Llama-3 PPL : {baseline_ppl:.4f}")
    print(f"Trained Engram PPL   : {trained_ppl:.4f}")
    
    diff = baseline_ppl - trained_ppl
    if diff > 0:
        print(f"SUCCESS: PPL improved by {diff:.4f}")
    else:
        print(f"FAIL: PPL degraded by {abs(diff):.4f}")

if __name__ == "__main__":
    main()