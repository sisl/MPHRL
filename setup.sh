#!/usr/bin/env bash
sudo apt-get update \
    && sudo apt-get install -y \
        htop \
        git \
        wget \
        cmake \
        pkg-config \
        build-essential \
        autoconf \
        curl \
        libtool \
        unzip \
        flex \
        bison \
        libgl1-mesa-dev \
        libgl1-mesa-glx \
        libglew-dev \
        libosmesa6-dev \
        xpra \
        xserver-xorg-dev \
        python-pip \
        python3-pip \
        python3-tk \
    && sudo apt-get clean \
    && wget https://www.roboti.us/download/mjpro150_linux.zip -O ../mujoco.zip \
    && unzip ../mujoco.zip -d ~/.mujoco \
    && sudo rm ../mujoco.zip \
    && sudo curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && sudo chmod +x /usr/local/bin/patchelf \
    && pip3 install \
        flatbuffers \
        ray \
        tensorflow \
        gym \
        psutil \
        imageio \
        mujoco-py \
    && sudo cp mjkey.txt ~/.mujoco/ \
    && export LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH} \
    && python3 -c 'import mujoco_py' \
    && python3 -c 'import gym; gym.make("HalfCheetah-v2")' \
    && sudo pip install git+https://github.com/deepmind/dm_control.git --user \
    && sudo pip3 install git+https://github.com/deepmind/dm_control.git --user \
    && sudo pip3 install --upgrade pip --user \
    && sudo python3 -m pip uninstall pip -y \
    && sudo apt install python3-pip --reinstall \
    && sudo python -m pip uninstall pip -y \
    && sudo apt install python-pip --reinstall \
    && sudo pip install git+https://github.com/rejuvyesh/gym-dmcontrol.git --user \
    && sudo pip3 install git+https://github.com/rejuvyesh/gym-dmcontrol.git --user \
    && sudo pip3 uninstall tensorflow protobuf -y \
    && sudo pip3 install tensorflow==1.8.0 \
    && sudo pip3 install yagmail \
    && touch screenlog.0 \
    && mkdir results \
    && sudo chmod -R 777 ~/.mujoco/ \
    && sudo pip3 uninstall -y dm_control \
    && sudo pip uninstall -y dm_control \
    && echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-384
    export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.5/dist_obv-packages:$HOME/.local/lib/python3.5/site-packages
    screen -wipe
    cd LifelongLearning
    clear
    screen -ls" >> ~/.bashrc \
    && make pip
