#!/bin/bash
#

############################################################################################
# install NVIDIA-driver/CUDA/CuDNN/Python/virtualenv/PyTorch on Windows 11
# Reference:
#    https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html
#    https://cloud.baidu.com/article/3326019
#    https://www.nvidia.com/en-us/geforce/graphics-cards/gt-1030/specifications/
#    https://blog.csdn.net/qq_45792697/article/details/123511064
#
# NOTE: Due to lack of compatibility for Tensorflow on new CUDA version (e.g. CUDA-12.6 and later).
#       A workable version set for both TF & PyTorch as follows:
#
#       Nvidia driver: no limit (newer the better)
#       CUDA: 11.7.0
#       Cudnn: 8.6.0
#       Tensorflow-gpu: 2.11.0
#       PyTorch: 2.2.0+cu121
#
# download Nvidia driver(need auth), then double click to install
wget https://cn.download.nvidia.com/Windows/572.83/572.83-desktop-win10-win11-64bit-international-dch-whql.exe
# launch cmd, then run "nvidia-smi" to verify the driver installation
nvidia-smi
#
# download CUDA, then double click to install (choose "custom install", and only select needed components,
# don't install Nvidia driver here)
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.76_windows.exe
# launch cmd, then run "nvcc --version" to verify the CUDA installation
nvcc --version
#
# download cuDNN, then extract the .zip package, and copy the "bin", "include", "lib" dir to corresponding dir
# under CUDA install path, like: "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\"
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/cudnn-windows-x86_64-9.8.0.87_cuda12-archive.zip
# launch cmd and cd to "CUDA_install_path\extras\demo_suite" dir, then run "bandwidthTest.exe" and "deviceQuery.exe"
# to verify the cuDNN installation
C:
cd Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\extras\demo_suite
bandwidthTest.exe
deviceQuery.exe
#
# download Python 3.13.3 from "https://www.python.org/", then double click to install,
# remember to choose "Add python.exe to PATH" on 1st page
wget https://www.python.org/ftp/python/3.13.3/python-3.13.3-amd64.exe
# launch cmd, then run "python" to verify the python installation
python
#
# launch cmd and install virtualenvwrapper for Windows, then create & config the virtualenv path
# See: https://blog.csdn.net/qq_45792697/article/details/123511064
pip install --upgrade pip
pip install virtualenvwrapper-win
#
# launch cmd to create a virtual environment, then install PyTorch-gpu 2.6.0 from "https://pytorch.org/"
mkvirtualenv -p python3.13 py313_ml
workon py313_ml
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
# run python, then import PyTorch to check if GPU is available
python
>>> import torch
>>> torch.cuda.is_available()
True


############################################################################################
# install NVIDIA-driver on Ubuntu 24.04
# Reference: https://zhuanlan.zhihu.com/p/1894453286439982079

# add nvidia driver repo into apt sources
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# install dependencies
sudo apt install -y alsa-utils
sudo apt install -y pciutils ubuntu-drivers-common

# check NVIDIA card info
lspci | grep -i nvidia
02:00.0 VGA compatible controller: NVIDIA Corporation GM107GL [Quadro K620] (rev a2)
02:00.1 Audio device: NVIDIA Corporation GM107 High Definition Audio Controller [GeForce 940MX] (rev a1)

# search recommended driver
sudo ubuntu-drivers devices
== /sys/devices/pci0000:00/0000:00:1a.0/0000:02:00.0 ==
modalias : pci:v000010DEd000013BBsv0000103Csd00001098bc03sc00i00
vendor   : NVIDIA Corporation
model    : GM107GL [Quadro K620]
driver   : nvidia-driver-555 - third-party non-free recommended
driver   : nvidia-driver-390 - third-party non-free
driver   : nvidia-driver-470 - third-party non-free
......
driver   : xserver-xorg-video-nouveau - distro free builtin
......

# install recommended driver, then reboot
sudo apt install nvidia-driver-555
sudo reboot

# after reboot, check installation with 'nvidia-smi' command
nvidia-smi

# the following CUDA/CuDNN & related tools installation is same as next part:
# "install NVIDIA-driver/CUDA/CuDNN on Ubuntu 22.04"



############################################################################################
# install NVIDIA-driver/CUDA/CuDNN on Ubuntu 22.04
# Reference: https://blog.csdn.net/qq_49323609/article/details/130310522
wget https://cn.download.nvidia.com/XFree86/Linux-x86_64/570.124.04/NVIDIA-Linux-x86_64-570.124.04.run
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/550.67/NVIDIA-Linux-x86_64-550.67.run
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz

sudo apt install -y gcc make cmake wget
sudo apt purge nvidia*
sudo cat >> /etc/modprobe.d/blacklist.conf << EOF

blacklist nouveau
options nouveau modeset=0
EOF

# latest default linux kernel version (6.5.0 and later) is built with gcc-12,
# so you need to switch system gcc to 12 to build & install NVIDIA driver, then
# switch back after install
sudo apt install gcc-12 g++-12
sudo ln -s -f /usr/bin/gcc-12 /usr/bin/gcc
sudo ln -s -f /usr/bin/g++-12 /usr/bin/g++

# uninstall any existing NVIDIA driver, then reboot
sudo apt-get --purge remove nvidia*
sudo update-initramfs -u
sudo reboot

# install NVIDIA driver
telinit 3
sudo service gdm stop
sudo service lightdm stop
# if you still meet install failure, try to download latest NVIDIA driver from
# https://www.nvidia.cn/drivers/lookup/ and use it
sudo chmod a+x NVIDIA-Linux-x86_64-550.67.run
sudo ./NVIDIA-Linux-x86_64-550.67.run -no-x-check -no-nouveau-check -no-opengl-files

# switch back gcc/g++
sudo ln -s -f /usr/bin/gcc-11 /usr/bin/gcc
sudo ln -s -f /usr/bin/g++-11 /usr/bin/g++
sudo reboot


# install CUDA, but need to disable drvier install here in menu, like:
#
# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ CUDA Installer                                                               │
# │ - [ ] Driver                                                                 │
# │      [ ] 515.43.04                                                           │
# │ + [X] CUDA Toolkit 11.7                                                      │
# │   [X] CUDA Demo Suite 11.7                                                   │
# │   [X] CUDA Documentation 11.7                                                │
# │   Options                                                                    │
# │   Install                                                                    │
# │                                                                              │
# └──────────────────────────────────────────────────────────────────────────────┘
#
sudo ./cuda_11.7.0_515.43.04_linux.run --override

# add CUDA related path in system
sudo cat >> ~/.bashrc << EOF

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
EOF

source ~/.bashrc


# install CuDNN
tar xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
cp -drf cudnn-linux-x86_64-8.6.0.163_cuda11-archive/include/* /usr/local/cuda/include/
cp -drf cudnn-linux-x86_64-8.6.0.163_cuda11-archive/lib/* /usr/local/cuda/lib64/
cp -drf cudnn-linux-x86_64-8.6.0.163_cuda11-archive/LICENSE /usr/local/cuda/


# install & create python3 virtualenv
sudo apt install -y python3-virtualenv
sudo pip install virtualenvwrapper

sudo cat >> ~/.bashrc << EOF
export VIRTUALENV_USE_DISTRIBUTE=1
export WORKON_HOME=$HOME/.virtualenvs
if [ -e $HOME/.local/bin/virtualenvwrapper.sh ];then
      source $HOME/.local/bin/virtualenvwrapper.sh
else if [ -e /usr/local/bin/virtualenvwrapper.sh ];then
            source /usr/local/bin/virtualenvwrapper.sh
     fi
fi
export PIP_VIRTUALENV_BASE=$WORKON_HOME
export PIP_RESPECT_VIRTUALENV=true
EOF

source ~/.bashrc
mkvirtualenv -p python3 py3ml
pip install --upgrade pip



# install some GPU monitor tools
sudo apt install nvtop
pip install nvitop
# Reference: https://cloud.tencent.com/developer/article/2260244
sudo apt install git qtdeclarative5-dev cmake
sudo git clone https://github.com/congard/nvidia-system-monitor-qt
cd nvidia-system-monitor-qt
sudo install icon.png /usr/share/icons/hicolor/512x512/apps/nvidia-system-monitor-qt.png
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DIconPath=/usr/share/icons/hicolor/512x512/apps/nvidia-system-monitor-qt.png -G "Unix Makefiles" ..
cd ..
cmake --build build --target qnvsm -- -j 4
sudo install build/qnvsm /usr/local/bin
qnvsm &


# install wenet dependency packages
sudo apt install -y sox libsox-dev
pip install numpy typing_extensions Pillow pyyaml sentencepiece tensorboard tensorboardX textgrid pytest mccabe cpplint tqdm deepspeed librosa langid sox
pip install torch==2.0.0 torchaudio==2.0.1 torchvision==0.15.1


# prepare wenet repo and launch demo training
# Reference: https://github.com/wenet-e2e/wenet/blob/main/docs/tutorial_aishell.md
git clone https://github.com/wenet-e2e/wenet.git
cd wenet/example/aishell/s0


# update run.sh to change data path, and train u2pp_conformer model for streaming inference
sed -i "/data=\/export\/data\/asr-data\/OpenSLR\/33\//c data=\/root\/aishell_data\/" run.sh
sed -i "/train_config=conf\/train_conformer.yaml/c train_config=conf/train_u2++_conformer.yaml" run.sh
sed -i "/dir=exp\/conformer/c dir=exp\/u2pp_conformer" run.sh

# change batch_size to 12 if you don't have enough GPU memory (batch_size==8 will cause bad convergence performance!!!)
sed -i "/        batch_size: 16/c         batch_size: 12" conf/train_u2++_conformer.yaml

# clean "data" & "exp/u2pp_conformer" dir to remove any legacy data
rm -rf data exp/u2pp_conformer

# stage -1: download & extract aishell data
bash run.sh --stage -1 --stop_stage -1
# stage 0: prepare training data
bash run.sh --stage 0 --stop_stage 0
# stage 1: extract optinal cmvn features
bash run.sh --stage 1 --stop_stage 1
# stage 2: generate label token dictionary
bash run.sh --stage 2 --stop_stage 2
# stage 3: prepare WeNet data format
bash run.sh --stage 3 --stop_stage 3
# stage 4: model training
bash run.sh --stage 4 --stop_stage 4


# install Chrome on Ubuntu as web browser, and use tensorboard to monitor training status
sudo wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i ./google-chrome-stable_current_amd64.deb
# update "/usr/share/applications/google-chrome.desktop" to allow GUI launch with root account
sed -i "/Exec=\/usr\/bin\/google-chrome-stable %U/c Exec=\/usr\/bin\/google-chrome-stable --no-sandbox %U" /usr/share/applications/google-chrome.desktop
sed -i "/Exec=\/usr\/bin\/google-chrome-stable --incognito/c Exec=\/usr\/bin\/google-chrome-stable --no-sandbox --incognito" /usr/share/applications/google-chrome.desktop

# if you launch Chrome from CLI with root, still need to add "--no-sandbox"
google-chrome-stable --no-sandbox

# launch tensorboard, web page at http://localhost:6006/
tensorboard --logdir=tensorboard/u2pp_conformer/



# export trained checkpoint & related config file, and convert to onnx for deployment
mkdir trained_checkpoint
cp -drf exp/u2pp_conformer/epoch_360.pt trained_checkpoint/
cp -drf exp/u2pp_conformer/train.yaml trained_checkpoint/
cp -drf data/train/global_cmvn trained_checkpoint/
cp -drf data/dict/lang_char.txt trained_checkpoint/words.txt

