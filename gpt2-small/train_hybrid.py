import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from engram_gpt2 import HybridEngramGPT2, engram_cfg

# 数据集类
class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=1024):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.examples = []
        tokens = tokenizer.encode(text)
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
    
    # 1. Load Model
    model = HybridEngramGPT2().to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # 2. Load Data
    tokenizer = AutoTokenizer.from_pretrained(engram_cfg.tokenizer_name_or_path)
    train_path = "/data2/home/wanghaoyi/data/wikitext/wiki.train.raw"
    dataset = TextDataset(train_path, tokenizer)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler) # A100 batch可以大一点
    
    # 3. Optimizer (Only optimize requires_grad=True params)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    
    if local_rank == 0:
        print("🚀 Start Training Engram on GPT-2 Backbone...")
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Params: {total_params/1e6:.2f}M | Trainable (Engram): {trainable_params/1e6:.2f}M")

    # 4. Loop
    model.train()
    for epoch in range(3): # 训练 3 个 epoch
        sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            batch = batch.to(device)
            loss, _ = model(batch, labels=batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0 and local_rank == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")
                
    if local_rank == 0:
        print("Saving Model...")
        # 只保存 Engram 部分的权重，或者保存整个 State Dict
        torch.save(model.module.state_dict(), "hybrid_engram_gpt2.pt")

if __name__ == "__main__":
    train()