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
        
After editing the .bashrc folder if there is some error regarding the path, the "same lines" can be used temporarily(every time the docker is restarted it needs to be executed so prevent errors.

4. Install the hiddenlayer for nnUNet
  
        pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git

The Folders should be structured as follows 

                 --------------------------------nnUNetFrame-----------------------------------
                 |                                                                            |
        ------Dataset-----------------------                                                nnUNet
        |                |                 |
    nnUNet_raw   nnUNet_preprocessed  nnUNet_result
 
 
# Preprocessing the data for training

1. download the raw dataset

2. Preprocess the raw data using the command -

        nnUNetv2_plan_and_preprocess -d <DATASET_ID>--verify_dataset_integrity

# To Train the preprocessed dataset
It is better to reduce the batch size to avoid the error "cuda: Out of memory"
There are 1,2,3,4,5 folds in training, where 0 is the least and so on.

-> For Training 2d data:
        
        nnUNet train <DATASET_NAME OR DATASET_ID> 2d <FOLD>
        
-> For Training 3d Data:

        nnUNet train <DATASET_NAME OR DATASET_ID> 3d_fullres <FOLD>
        
# How to use the files, what changes have been made/to be made
All the Dataset_generator python files are created for nnUNetv2 
  1. BraTS2021- The dataset can be downloaded from official website or from kaggle is the datasset is not available on th official website. 
 
