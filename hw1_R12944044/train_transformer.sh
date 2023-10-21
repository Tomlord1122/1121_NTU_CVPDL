cd detr
export CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch\
 --nproc_per_node=1 \
  --use_env main.py \
  --coco_path /local/tomlord1122/1121-CVPDL/detr/hw1_dataset\
  --epoch 350 \
  --output_dir="output" \
  --resume="./detr-r50_8.pth