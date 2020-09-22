#!/bin/bash

chmod 1777 /tmp
apt update --allow-unauthenticated

apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git

unset PYTHONPATH
export DEBIAN_FRONTEND=noninteractive

apt-get install -y \
    graphviz \
    texlive-latex-base \
    texlive-latex-extra

pip install --upgrade pip setuptools

pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

pip install tensorflow==2.3.0

pip install git+https://github.com/fastestimator/fastestimator.git

fastestimator train mnist_tf.py --save_dir $SM_MODEL_DIR --summary tf_mnist

#if you have specific training data, make use the $SM_CHANNEL_TRAINING
