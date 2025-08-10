

## 1.1 Install basic system packages

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
