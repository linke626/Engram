import torch
import pandas as pd
import os
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from engram_gpt2 import HybridEngramGPT2, engram_cfg

# MMLU 题目格式化模板 (0-shot)
def format_prompt(row):
    prompt = f"Question: {row[0]}\n"
    prompt += f"A. {row[1]}\n"
    prompt += f"B. {row[2]}\n"
    prompt += f"C. {row[3]}\n"
    prompt += f"D. {row[4]}\n"
    prompt += "Answer:"
    return prompt

def evaluate_model(model, tokenizer, df, device):
    model.eval()
    correct = 0
    total = len(df)
    
    choices = [" A", " B", " C", " D"]
    choice_ids = [tokenizer.encode(c)[0] for c in choices]
    
    print(f"开始评测 {total} 道题目...")
    
    with torch.no_grad():
        for index, row in df.iterrows():
            prompt = format_prompt(row)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            # --- 修改开始: 兼容不同的模型输出格式 ---
            outputs = model(inputs.input_ids)
            
            if isinstance(outputs, torch.Tensor):
                # 情况1: Hybrid 模型直接返回了 Tensor
                logits = outputs
            elif isinstance(outputs, tuple):
                # 情况2: 旧版 HF 模型返回 tuple (loss, logits)
                logits = outputs[1]
            else:
                # 情况3: 新版 HF 模型返回 Output 对象 (拥有 .logits 属性)
                logits = outputs.logits
            # --- 修改结束 ---

            # 取最后一个 token 的 logits
            last_token_logits = logits[0, -1, :]
            choice_logits = last_token_logits[choice_ids]
            predicted_idx = torch.argmax(choice_logits).item()
            predicted_char = ["A", "B", "C", "D"][predicted_idx]
            
            label_char = row[5]
            if predicted_char == label_char:
                correct += 1
            
            if index < 3:
                print(f"题目: {row[0][:30]}... | 预测: {predicted_char} | 正确: {label_char}")

    accuracy = correct / total
    return accuracy

def run_mmlu_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path)
    
    # 设定测试科目：global_facts (比较简单，适合测试)
    # 你可以改成其他科目，如 'elementary_mathematics_test.csv'
    subject = "global_facts"
    csv_path = f"/data2/home/wanghaoyi/data/mmlu/data/test/{subject}_test.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到文件: {csv_path}")
        # 尝试寻找上一级目录（有些解压结构不同）
        csv_path = f"/data2/home/wanghaoyi/data/mmlu/data/{subject}_test.csv"
        if not os.path.exists(csv_path):
            print("请检查 MMLU 数据路径")
            return

    # Pandas 读取 CSV (无 header)
    df = pd.read_csv(csv_path, header=None)
    print(f"\n=== 正在评测科目: {subject} (共 {len(df)} 题) ===")

    # 1. 评测 Baseline
    print("\n--- Baseline (Frozen GPT-2) ---")
    baseline = AutoModelForCausalLM.from_pretrained(engram_cfg.tokenizer_name_or_path).to(device)
    acc_base = evaluate_model(baseline, tokenizer, df, device)
    print(f"Baseline Accuracy: {acc_base:.2%}")
    del baseline

    # 2. 评测 Hybrid Engram
    print("\n--- Hybrid (GPT-2 + Engram) ---")
    model = HybridEngramGPT2().to(device)
    state_dict = torch.load("hybrid_engram_gpt2.pt", map_location=device)
    model.load_state_dict(state_dict)
    
    acc_ours = evaluate_model(model, tokenizer, df, device)
    print(f"Hybrid Accuracy:   {acc_ours:.2%}")

    print(f"\n差距 (Hybrid - Base): {acc_ours - acc_base:.2%}")

if __name__ == "__main__":
    run_mmlu_test()
