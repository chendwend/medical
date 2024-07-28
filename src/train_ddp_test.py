import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import pickle

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def broadcast_object(obj, master_process):
    device = f'cuda:{int(os.environ["LOCAL_RANK"])}'
    if master_process:
        # Only the source process executes this block
        obj_bytes = pickle.dumps(obj)
        obj_size = torch.tensor(len(obj_bytes), dtype=torch.long, device=device)
    else:
        obj_size = torch.tensor(0, dtype=torch.long, device=device)

    # Broadcast the size of the byte stream to all processes
    print("before broadcast 1")
    dist.broadcast(obj_size, 0)

    # Allocate buffer for the object's byte stream
    obj_bytes = bytearray(obj_size.item())

    if master_process:
        # Only the source fills the byte buffer
        obj_bytes[:] = pickle.dumps(obj)

    # Create a tensor wrapper for the byte buffer for broadcasting
    obj_tensor = torch.ByteTensor(obj_bytes).to(device)
    # Broadcast the byte stream
    print("before broadcast 2")
    dist.broadcast(obj_tensor, 0)

    # Deserialize the byte stream back into the Python object
    obj = pickle.loads(obj_tensor.cpu().numpy().tobytes())
    torch.distributed.barrier() 
    return obj




def setup():
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")

def cleanup():
    dist.destroy_process_group()

def train():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    setup()
    
    if rank == 0:
        wandb.init()
        config = wandb.config
        hp_dict = {k: v for k, v in dict(config).items()}
    else:
        hp_dict = None
    
    # Broadcast config from rank 0 to all processes
    # config = dist.broadcast_object_list([config], src=0)[0]
    hp_dict = broadcast_object(hp_dict, rank==0)
    print(f"{rank}: {hp_dict}")
    torch.cuda.set_device(rank)
    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=hp_dict['lr'])

    for epoch in range(hp_dict['epochs']):
        # Simulated data
        inputs = torch.randn(20, 10).to(rank)
        targets = torch.randn(20, 1).to(rank)

        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        if rank == 0:
            wandb.log({"loss": loss.item(), "epoch": epoch})

    cleanup()

if __name__ == "__main__":
    train()