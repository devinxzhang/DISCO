export PYTHONPATH="./"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=29501 --use_env tools/train_mm_ori2.py --cfg configs/deliver_dinov2.yaml
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29501 --use_env tools/train_mm_ori2.py --cfg configs/mcubes_rgbadn.yaml
# CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.launch --nproc_per_node=1 --master_port=29502 --use_env tools/train_mm_ori2.py --cfg configs/muses_dinov2.yaml
# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tools/train_mm.py --cfg configs/deliver_dinov2.yaml
# 
# CUDA_VISIBLE_DEVICES=5 python tools/val_mm.py --cfg configs/mcubes_rgbadn.yaml

