import os
import pickle

import torch
import torch.distributed as dist


def ddp_setup(backend="nccl"):
    if backend == "nccl" and torch.distributed.is_nccl_available():
        dist.init_process_group(backend)
        
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        master_process = global_rank == 0
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        print(f"GPU {local_rank} - Using device: {device}")
    else:
        raise("nccl backend isn't available or wasn't selected")
    return global_rank, local_rank, world_size, master_process, device

def broadcast_object(obj, master_process:bool):
    """Broadcast object from master process to all processes.

    Args:
        obj (_type_): _description_
        master_process (_type_): _description_

    Returns:
        _type_: _description_
    """

    device = f'cuda:{int(os.environ["LOCAL_RANK"])}'
    if master_process:
        obj_bytes = pickle.dumps(obj)
        obj_size_tensor = torch.tensor(len(obj_bytes), dtype=torch.long, device=device)
    else:
        obj_size_tensor = torch.tensor(0, dtype=torch.long, device=device)

    dist.broadcast(obj_size_tensor, 0)
    # obj_bytes = bytearray(obj_size.item())

    # if master_process:
    #     # Only the source fills the byte buffer
    #     obj_bytes[:] = pickle.dumps(obj)
    if master_process:
        obj_bytes_tensor = torch.ByteTensor(list(obj_bytes)).to(device)
    else:
        obj_bytes_tensor = torch.empty(obj_size_tensor, dtype=torch.uint8).to(device)

    dist.broadcast(obj_bytes_tensor, src=0)

    if not master_process:
        # Deserialize the byte stream back into the Python object
        obj_bytes = obj_bytes_tensor.cpu().numpy().tobytes()
        obj = pickle.loads(obj_bytes)
    
    # torch.distributed.barrier() 
    return obj

def ddp_cleanup() -> None:
    dist.destroy_process_group()
    torch.cuda.empty_cache()