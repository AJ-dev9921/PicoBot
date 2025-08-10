1.1 Install basic system packages
bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git wget curl unzip zip \
    python3 python3-pip python3-venv libglib2.0-0
//
These are needed for Python, OpenCV, etc.
