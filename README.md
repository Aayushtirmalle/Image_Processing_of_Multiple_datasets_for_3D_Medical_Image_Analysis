# nnUNetv2
Training the nnUNetv2 using virtual machine and Docker

# Setting up the software for training

1. Install Pytorch, cuda

1. Install the nnUNet software using interactive shell

        git clone https://github.com/MIC-DKFZ/nnUNet.git
        cd nnUNet
        pip install -e .


2. Create folders "nnUNet_raw, nnUNet_preprocessed, nnUNet_trained data" and set the path for the same, while using docker set the paths with respect to the docker path.

Locate the .bashrc file in your home folder of the docker and add the following lines to the bottom:

        export nnUNet_raw="/home/abc/nnUNetFrame/dataset/nnUNet_raw"
        export nnUNet_preprocessed="/home/abc/nnUNetFrame/dataset/nnUNet_preprocessed"
        export nnUNet_results="/home/abc/nnUNetFrame/dataset/nnUNet_results"


4. Install the hiddenlayer for nnUNet
  
        pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git

# Preprocessing the data for training

1. download the raw dataset

2. Preprocess the raw data



 
