import torch
import math
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# 导入你的模型定义
from engram_gpt2 import HybridEngramGPT2, engram_cfg

# ================= 配置 =================
# 测试集路径 (根据你的 ls 结果修改)
TEST_FILE_PATH = "../../data/wikitext/wiki.test.raw" 
# 训练好的权重文件
CHECKPOINT_PATH = "hybrid_engram_gpt2.pt"
# Batch Size (评测时不需要反向传播，显存占用小，可以调大)
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 数据集类 (复用) =================
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=1024):
        print(f"Loading test data from {file_path}...")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # 简单的切分逻辑
        self.examples = []
        tokens = tokenizer.encode(text)
        total_tokens = len(tokens)
        
        # 丢弃最后不足 block_size 的部分，保证整齐
        for i in range(0, total_tokens - block_size + 1, block_size):
            self.examples.append(tokens[i:i+block_size])
            
        print(f"✅ Loaded {len(self.examples)} blocks (Context Length: {block_size}).")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)

# ================= 核心评测函数 =================
def calculate_perplexity(model, dataloader, device, description="Evaluating"):
    model.eval()
    total_loss = 0.0
    total_steps = 0
    
    progress_bar = tqdm(dataloader, desc=description)
    
    with torch.no_grad():
        for batch in progress_bar:
            batch = batch.to(device)
            # labels=batch 意味着计算标准的语言模型 Loss
            loss, _ = model(batch, labels=batch)
            
            total_loss += loss.item()
            total_steps += 1
            
            # 实时显示当前 PPL
            current_ppl = math.exp(total_loss / total_steps)
            progress_bar.set_postfix({'Avg Loss': f"{total_loss/total_steps:.4f}", 'PPL': f"{current_ppl:.2f}"})

    avg_loss = total_loss / total_steps
    perplexity = math.exp(avg_loss)
    return perplexity

# ================= 主程序 =================
def main():
    print(f"Using device: {DEVICE}")
    
    # 1. 准备 Tokenizer 和 数据加载器
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path)
    test_dataset = TextDataset(TEST_FILE_PATH, tokenizer, block_size=1024)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. 初始化模型 (随机初始化的 Engram + 冻结的 GPT-2)
    print("\n[Phase 1] Initializing Model (Pre-training state)...")
    model = HybridEngramGPT2()
    model.to(DEVICE)
    
    # 3. 评测训练前 (Baseline)
    print("\n>>> Calculating Pre-training Perplexity...")
    ppl_pre = calculate_perplexity(model, test_loader, DEVICE, description="Pre-train Eval")
    print(f"Result: Pre-training PPL = {ppl_pre:.4f}")
    
    # 4. 加载训练好的权重
    print(f"\n[Phase 2] Loading weights from {CHECKPOINT_PATH}...")
    if os.path.exists(CHECKPOINT_PATH):
        # 注意：因为之前的保存逻辑可能是 model.module.state_dict() (DDP) 或者 model.state_dict()
        # 我们尝试智能加载
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        # 处理 DDP 保存时多出来的 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        # strict=False 因为 GPT-2 部分是冻结的且可能有些 buffer 差异，
        # 但主要我们想加载 Engram 的权重。
        # 如果你只保存了 Engram 部分，这里加载会更复杂。
        # 假设你保存的是整个 model.state_dict()
        keys = model.load_state_dict(new_state_dict, strict=False)
        print(f"Weights loaded. Missing keys (should be none or minor): {len(keys.missing_keys)}")
    else:
        print(f"❌ Error: Checkpoint {CHECKPOINT_PATH} not found!")
        return

    # 5. 评测训练后
    print("\n>>> Calculating Post-training Perplexity...")
    ppl_post = calculate_perplexity(model, test_loader, DEVICE, description="Post-train Eval")
    print(f"Result: Post-training PPL = {ppl_post:.4f}")
    
    # 6. 总结
    print("\n" + "="*40)
    print("Evaluation Summary on WikiText-Test")
    print("="*40)
    print(f"Before Training PPL: {ppl_pre:.4f}")
    print(f"After Training  PPL: {ppl_post:.4f}")
    delta = ppl_pre - ppl_post
    print(f"Improvement:         {delta:.4f} points")
    print("="*40)

if __name__ == "__main__":
    main()