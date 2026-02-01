import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram_gpt2 import HybridEngramGPT2, engram_cfg

def calc_ppl(model, test_path, tokenizer, device, stride=512):
    model.eval()
    with open(test_path, "r", encoding="utf-8") as f:
        text = f.read()[:50000] # 只测前5万字符以快速验证
    
    encodings = tokenizer(text, return_tensors="pt")
    max_length = 1024
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0
    
    print(f"Calculating PPL on {seq_len} tokens...")
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # 兼容 HuggingFace Output 和 Tuple Output
            loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
            nlls.append(loss * trg_len)
        
        prev_end_loc = end_loc
        if end_loc == seq_len: break
        
    return torch.exp(torch.stack(nlls).sum() / end_loc).item()

def run_benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path)
    test_path = "/data2/home/wanghaoyi/data/wikitext/wiki.test.raw" # 使用测试集
    
    # 1. Baseline
    print("--- Evaluating Baseline (Frozen GPT-2) ---")
    baseline = AutoModelForCausalLM.from_pretrained(engram_cfg.tokenizer_name_or_path).to(device)
    ppl_base = calc_ppl(baseline, test_path, tokenizer, device)
    print(f"Baseline PPL: {ppl_base:.2f}")
    del baseline
    
    # 2. Ours
    print("\n--- Evaluating Hybrid (GPT-2 + Engram) ---")
    model = HybridEngramGPT2().to(device)
    # 加载你训练好的权重
    state_dict = torch.load("hybrid_engram_gpt2.pt", map_location=device)
    model.load_state_dict(state_dict)
    
    ppl_ours = calc_ppl(model, test_path, tokenizer, device)
    print(f"Hybrid PPL:   {ppl_ours:.2f}")
    
    print(f"\nImprovement: {ppl_base - ppl_ours:.2f} points")

if __name__ == "__main__":
    run_benchmark()