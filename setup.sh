#!/bin/sh
#
# This script installs all Icarus dependencies
#
# It has been tested successfully only on Ubuntu 12.04+.
#
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

sudo apt-get install python3 ipython3 python3-pip python3-scipy python3-matplotlib python3-nose python3-sphinx python3-networkx 
sudo pip3 install -U fnss numpydoc
sudo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
