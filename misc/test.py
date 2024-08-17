import os
import time

import matplotlib
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from src.dataloaders import get_dataloaders

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def setup(rank, seed):
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    torch.manual_seed(seed) 

def cleanup():
    dist.destroy_process_group()
    torch.cuda.empty_cache()


rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
master_process = rank == 0

try:
    setup(rank, 42)
    torch.cuda.set_device(rank)
     train_loader, val_loader, test_loader = get_dataloaders(cfg["folders"][task], cfg.preprocessing.image_size, cfg.preprocessing.norm, hp_dict["batch_size"], world_size, rank, cfg.testing)

finally:
    cleanup()












class DummyDataset(Dataset):
    def __init__(self, size=50000, dim=1000):
        self.size = size
        self.dim = dim
        self.data = [torch.randn(dim) for _ in range(size)]
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Simulate some data processing
        return self.data[idx] + torch.randn(self.dim) * 0.1

def test_dataloader(num_workers, batch_size=64, num_epochs=3):
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Warmup
    for batch in dataloader:
        batch = batch.to(device)
        output = torch.nn.functional.relu(batch)
        break
    
    start_time = time.perf_counter()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            batch = batch.to(device)
            # Simulate more complex GPU processing
            output = torch.nn.functional.relu(torch.matmul(batch, batch.t()))
            torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    return total_time

def run_benchmark(max_workers=16, batch_size=64, num_epochs=3):
    results = []
    
    for num_workers in range(0, max_workers + 1, 1):
        times = []
        for _ in range(3):  # Run each test 3 times
            time_taken = test_dataloader(num_workers, batch_size, num_epochs)
            times.append(time_taken)
        avg_time = np.mean(times)
        results.append((num_workers, avg_time))
        print(f"num_workers: {num_workers}, avg_time: {avg_time:.4f} seconds")
    
    return results

# ... (plot_results function remains the same)

def plot_results(results):
    workers, times = zip(*results)
    
    plt.figure(figsize=(10, 6))
    plt.plot(workers, times, marker='o')
    plt.title('DataLoader Performance vs. Number of Workers')
    plt.xlabel('Number of Workers')
    plt.ylabel('Time (seconds)')
    plt.grid(True)
    # plt.savefig('num_workers.png')



if __name__ == "__main__":
    results = run_benchmark()
    # plot_results(results)
    
    optimal_workers = min(results, key=lambda x: x[1])[0]
    print(f"\nOptimal number of workers: {optimal_workers}")