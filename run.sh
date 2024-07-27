#!/bin/bash


export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=wlp4s0
export NCCL_IB_DISABLE=1      
export NCCL_NET_GDR_LEVEL=0     
export NCCL_LAUNCH_MODE=PARALLEL
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISABLE_ADDR2LINE=1
# export NCCL_NSOCKS_PERTHREAD=4


source /opt/anaconda3/bin/activate medical

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=12355 test.py