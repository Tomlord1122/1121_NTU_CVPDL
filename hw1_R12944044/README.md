# readme_test

1. 先將作業給的Dataset放入detr資料夾，並載入 https://github.com/facebookresearch/detr.git 所提供的 pre-trained model. （需自行下載），並按照此repo將Dataset更改符合coco format

2. 將此model透過detr的`change_class_num.py` 將model class做更改以符合作業dataset只有8個class。
3. 更改 models/detr.py 裡面的num_classes = 8
4. train model 

```bash
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --coco_path /local/tomlord1122/1121-CVPDL/detr/hw1_dataset --epoch 350 --output_dir="output" --resume="./detr-r50_8.pth
```

5. `[inference.py]` 將訓練好的model用validation image data測試並儲存為output.json
6. `[evaluate.py]` 測試結果 ( test directory 位置在 detr/）

```
python version = 3.10.12
asttokens==2.4.0
backcall==0.2.0
certifi==2023.7.22
charset-normalizer==3.3.0
comm==0.1.4
contourpy==1.1.1
cycler==0.12.1
debugpy==1.8.0
decorator==5.1.1
exceptiongroup==1.1.3
executing==2.0.0
filelock==3.12.4
fonttools==4.43.1
fsspec==2023.9.2
huggingface-hub==0.17.3
idna==3.4
ipykernel==6.25.2
ipython==8.16.1
jedi==0.19.1
Jinja2==3.1.2
jupyter_client==8.4.0
jupyter_core==5.4.0
kiwisolver==1.4.5
lightning-utilities==0.9.0
MarkupSafe==2.1.3
matplotlib==3.8.0
matplotlib-inline==0.1.6
mpmath==1.3.0
nest-asyncio==1.5.8
networkx==3.1
numpy==1.26.1
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.18.1
nvidia-nvjitlink-cu12==12.2.140
nvidia-nvtx-cu12==12.1.105
opencv-python==4.8.1.78
opencv-python-headless==4.8.1.78
packaging==23.2
parso==0.8.3
pexpect==4.8.0
pickleshare==0.7.5
Pillow==10.1.0
platformdirs==3.11.0
prompt-toolkit==3.0.39
psutil==5.9.6
ptyprocess==0.7.0
pure-eval==0.2.2
pycocotools==2.0.7
Pygments==2.16.1
pyparsing==3.1.1
python-dateutil==2.8.2
PyYAML==6.0.1
pyzmq==25.1.1
regex==2023.10.3
requests==2.31.0
safetensors==0.4.0
scipy==1.11.3
six==1.16.0
stack-data==0.6.3
supervision==0.15.0
sympy==1.12
tokenizers==0.14.1
torch==2.1.0
torchmetrics==1.2.0
torchvision==0.16.0
tornado==6.3.3
tqdm==4.66.1
traitlets==5.11.2
transformers==4.34.0
triton==2.1.0
typing_extensions==4.8.0
urllib3==2.0.6
wcwidth==0.2.8
```