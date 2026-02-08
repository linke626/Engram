import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
from torch.optim import AdamW
from engram_Llama3_8B_Instruct import HybridEngramLlama, engram_cfg

# ================= 硬件设置 =================
# 你指定使用卡号 4-7。
# 在 DDP 中，通常通过 `torchrun --nproc_per_node=4 ...` 启动。
# 如果你在外部没有设置 CUDA_VISIBLE_DEVICES，可以在这里强制设置，
# 但最好的做法是在命令行: CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun ...
# 这里假设外部已经做好了映射，或者我们只使用看到的设备。

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=1024):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at: {file_path}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        self.examples = []
        # Llama-3 needs explicit EOS/BOS handling usually, but simple encoding works for raw text
        tokens = tokenizer.encode(text, add_special_tokens=True)
        
        # Simply chunking
        for i in range(0, len(tokens) - block_size, block_size):
            self.examples.append(tokens[i:i+block_size])
        print(f"Loaded {len(self.examples)} blocks from {file_path}")
        
    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return torch.tensor(self.examples[i], dtype=torch.long)

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def train():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    # 1. Load Model (Use bfloat16 for Llama 3)
    # The model inside initializes in bf16 where appropriate
    model = HybridEngramLlama().to(device, dtype=torch.bfloat16)
    
    # DDP wrapper
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # 2. Load Data
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token # Llama usually uses eos_token as pad for training
        
    train_path = "/data2/home/wanghaoyi/data/wikitext/wiki.train.raw"
    
    if local_rank == 0:
        print(f"Reading data from {train_path}...")
        
    dataset = TextDataset(train_path, tokenizer, block_size=512) # Reduced block size for debug/VRAM safety
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler) # Adjust batch size based on VRAM (Llama-3 8B takes ~16GB in bf16)
    
    # 3. Optimizer (Only optimize requires_grad=True params)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    if local_rank == 0:
        print("🚀 Start Training Engram on Llama-3-8B Backbone...")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Params: {total_params/1e6:.2f}M | Trainable (Engram): {trainable_params/1e6:.2f}M")

    # 4. Loop

    # 计算总步数
    total_steps_per_epoch = len(dataloader)
    total_training_steps = total_steps_per_epoch * 3 # 假设 epoch 数为 3

    if local_rank == 0:
        print(f"Total steps per epoch: {total_steps_per_epoch}")
        print(f"Total training steps: {total_training_steps}")

    # 4. Loop
    model.train()
    for epoch in range(3): 
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            batch = batch.to(device)
            
            # Forward
            loss, _ = model(batch, labels=batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 修改打印日志，显示当前步数/总步数
            if step % 5 == 0 and local_rank == 0:
                print(f"Epoch {epoch+1}/3 | Step {step}/{total_steps_per_epoch} | Loss: {loss.item():.4f}")
    
    if local_rank == 0:
        print("Saving Engram Model...")
        state_dict = model.module.state_dict()
        # 只保存 Engram 相关的参数
        engram_state_dict = {k: v for k, v in state_dict.items() if "engram_module" in k}
        torch.save(engram_state_dict, "engram_llama3_weights.pt")

if __name__ == "__main__":
    train()