

##  🏋️ 1.1 Install basic system packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git wget curl unzip zip python3 python3-pip python3-venv libglib2.0-0
```
These are needed for Python, OpenCV, etc.

## 1.2 Install NVIDIA Driver
You confirmed you have:

    Driver version: 575.64.03

    CUDA version: 12.9

    GPU: GeForce GTX 1060 3GB

✅ Everything looks good.

## 1.3 Install Conda

Download & install Miniconda:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```
Restart terminal after install.

## 1.4 Create Python environment
```bash
conda create -n yolov8 python=3.10 -y
conda activate yolov8
```

## 1.5 Install PyTorch (CUDA 12.1 — compatible with your 12.9)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Test it:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
✅ It should print True.

## 1.6 Install YOLOv8
```bash
pip install ultralytics
```
### Test CLI:
```bash
yolo help
```

## 1.7 Edit data.yaml

Use full absolute paths. Example:
```bash
train: /home/afraj/yolov8-dataset/train/images
val: /home/afraj/yolov8-dataset/val/images

nc: 3  # change this to your number of classes
names: ['class_0', 'class_1', 'class_2']  # replace with real class names
```
To get full paths:
```bash
realpath yolov8-dataset/train/images
realpath yolov8-dataset/val/images
```

## 1.8 Train YOLOv8-nano

Run this command (adjust batch if you get OOM):
```bash
yolo task=detect mode=train model=yolov8n.pt \
  data=/home/afraj/yolov8-dataset/data.yaml \
  epochs=50 imgsz=640 batch=4 name=yolov8n_run
```
Training outputs saved to:

📁 runs/detect/yolov8n_run/
  ↳ best.pt ← this is the final model
  
✅ Training complete


## 📤 2. Export to TFLite (for Android)

First try direct export:
```bash
yolo export model=runs/detect/yolov8n_run/weights/best.pt format=tflite
```
It will produce:

📄 best.tflite
