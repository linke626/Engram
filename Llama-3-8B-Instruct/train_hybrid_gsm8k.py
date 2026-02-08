import os
import torch
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from torch.optim import AdamW
from engram_Llama3_8B_Instruct import HybridEngramLlama, engram_cfg

# ================= 硬件设置 =================
# ... (注释保持不变) ...

class GSM8KDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=1024):
        file_path = os.path.expanduser(file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at: {file_path}")
            
        print(f"Loading GSM8K data from {file_path}...")
        
        self.examples = []
        all_tokens = [] # 使用 list 暂存 tokens，比字符串拼接快得多
        
        # 1. 逐行读取并 Tokenize (更省内存，更安全)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    question = data.get('question', '').strip()
                    answer = data.get('answer', '').strip()
                    
                    if not question or not answer:
                        continue
                    
                    # 格式化: 加上 Llama-3 的特殊 Token 可能会更好，但这里保持简单文本格式
                    # 注意：我们在每道题后面加了 EOS token，帮助模型区分题目边界
                    text_sample = f"Question: {question}\nAnswer: {answer}\n{tokenizer.eos_token}\n"
                    
                    # 编码单条数据
                    sample_tokens = tokenizer.encode(text_sample, add_special_tokens=False)
                    all_tokens.extend(sample_tokens)
                    
                except json.JSONDecodeError:
                    continue

        # 2. 统一加上 BOS (如果需要)
        # Llama-3 通常需要 bos_token 在最开头
        if tokenizer.bos_token_id is not None:
             all_tokens = [tokenizer.bos_token_id] + all_tokens

        # 3. 切块 (Chunking)
        # 将巨大的 token 列表切分成 block_size 长度的小块
        total_tokens = len(all_tokens)
        if total_tokens < block_size:
            print(f"Warning: Data size ({total_tokens}) is smaller than block_size ({block_size}).")
        
        for i in range(0, total_tokens - block_size + 1, block_size):
            self.examples.append(all_tokens[i : i + block_size])
            
        print(f"Loaded {len(self.examples)} blocks from {file_path}")
        
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return torch.tensor(self.examples[i], dtype=torch.long)

def setup_ddp():
    if "LOCAL_RANK" not in os.environ:
        # 如果不是通过 torchrun 启动（单卡调试），做个兼容
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def train():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    # 1. Load Model
    # 确保 engram_Llama3_8B_Instruct.py 里的 Config 路径是对的
    model = HybridEngramLlama().to(device, dtype=torch.bfloat16)
    
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # 2. Load Data
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
        
    train_path = "~/data/gsm8k/train.jsonl" 
    
    if local_rank == 0:
        print(f"Reading data from {train_path}...")
        
    # block_size 设为 512 对于 8B 模型 + Engram 是比较安全的
    # 如果显存不够，可以降到 256
    dataset = GSM8KDataset(train_path, tokenizer, block_size=512)
    
    if len(dataset) == 0:
        if local_rank == 0: print("Error: Dataset is empty!")
        return

    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler) 
    
    # 3. Optimizer
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    if local_rank == 0:
        print("🚀 Start Training Engram on Llama-3-8B Backbone...")
        # ... (参数统计代码保持不变) ...

    # 4. Loop
    total_steps_per_epoch = len(dataloader)
    # 增加梯度累积步数，变相增大 Batch Size
    gradient_accumulation_steps = 4 
    total_training_steps = total_steps_per_epoch * 3 

    if local_rank == 0:
        print(f"Total steps per epoch: {total_steps_per_epoch}")
    
    model.train()
    for epoch in range(10): 
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            # Forward
            loss, _ = model(batch, labels=batch)
            loss = loss / gradient_accumulation_steps # Normalize loss
            
            loss.backward()
            
            # Gradient Accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if step % 10 == 0 and local_rank == 0:
                # 乘回去打印真实的 loss
                print(f"Epoch {epoch+1}/10 | Step {step}/{total_steps_per_epoch} | Loss: {loss.item() * gradient_accumulation_steps:.4f}")
    
    if local_rank == 0:
        print("Saving Engram Model...")
        state_dict = model.module.state_dict()
        engram_state_dict = {k: v for k, v in state_dict.items() if "engram_module" in k}
        torch.save(engram_state_dict, "engram_llama3_gsm8k.pt")
        
    dist.destroy_process_group() # Clean up

if __name__ == "__main__":
    train()