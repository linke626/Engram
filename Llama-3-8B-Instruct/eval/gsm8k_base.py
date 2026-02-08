import json
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= 配置区域 =================
MODEL_PATH = "/data2/home/wanghaoyi/models/Meta-Llama-3-8B-Instruct" 
DATA_PATH = "/data2/home/wanghaoyi/data/gsm8k/test.jsonl"
OUTPUT_PATH = "./result_gsm8k.jsonl"
BATCH_SIZE = 1
MAX_NEW_TOKENS = 512
# ===========================================

def load_model():
    print(f"正在加载模型: {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    # Llama 3 建议使用 bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
    )
    
    # 设置 pad token，防止报错
    tokenizer.pad_token = tokenizer.eos_token
    
    return tokenizer, model

def extract_answer_number(text):
    """
    从 GSM8K 的标准答案或模型输出中提取数字。
    """
    if not text: return None
    # 1. 尝试寻找 #### 后的内容
    if "####" in text:
        target = text.split("####")[-1].strip()
        return target.replace(",", "")
    
    # 2. 兜底策略：提取最后一个数字
    numbers = re.findall(r'-?\d+\.?\d*', text.replace(",", ""))
    if numbers:
        return numbers[-1]
    return None

def main():
    tokenizer, model = load_model()
    
    # Llama 3 的特殊停止符
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # 读取数据
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"共加载 {len(lines)} 条测试数据。")
    
    correct_count = 0
    total_processed = 0

    # 写入文件（追加模式或覆盖模式，这里用覆盖 'w'）
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as out_f:
        for line in tqdm(lines):
            try:
                problem_data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            question = problem_data['question']
            ground_truth_full = problem_data['answer']
            
            ground_truth_val = extract_answer_number(ground_truth_full)

            messages = [
                {"role": "system", "content": "You are a helpful assistant. Solve the math problem step by step. Put the final answer after '####'."},
                {"role": "user", "content": question}
            ]
            
            input_ids = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=MAX_NEW_TOKENS,
                    eos_token_id=terminators,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = outputs[0][input_ids.shape[-1]:]
            decoded_output = tokenizer.decode(response, skip_special_tokens=True)
            
            model_val = extract_answer_number(decoded_output)
            
            # 判断正确性
            is_correct = False
            if model_val and ground_truth_val:
                try:
                    # 允许 42.0 == 42
                    if float(model_val) == float(ground_truth_val):
                        is_correct = True
                except ValueError:
                    is_correct = (model_val == ground_truth_val)

            if is_correct:
                correct_count += 1
            total_processed += 1

            result_item = {
                "question": question,
                "ground_truth": ground_truth_full,
                "model_output": decoded_output,
                "extracted_truth": ground_truth_val,
                "extracted_model": model_val,
                "is_correct": is_correct
            }
            
            out_f.write(json.dumps(result_item, ensure_ascii=False) + "\n")
            out_f.flush()

    print(f"\n测试结束！")
    if total_processed > 0:
        print(f"最终准确率: {(correct_count/total_processed)*100:.2f}%")
    else:
        print("未处理任何数据。")
    print(f"结果已保存至: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()