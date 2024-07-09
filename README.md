# Image Processing of Multiple datasets for 3D Medical Image Analysis
## Using nnUNetv2
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
        
        nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD [--npz]
        
-> For Training 3d Data:

        nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD [--npz]
        

 
# Medical Image Analysis with nnUNetv2 and Alzheimer's Disease Detection

This repository provides the code and instructions to perform two key tasks in medical image analysis:
1. Biomedical image segmentation using nnUNetv2.
2. Alzheimer's disease detection using a deep learning model trained on structural MRIs.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Software 1: nnUNetv2 for Biomedical Image Segmentation](#Software-1-nnunetv2-for-biomedical-image-segmentation)
  - [Datasets](#datasets)
  - [Data Preprocessing](#data-preprocessing)
  - [Training the Model](#training-the-model)
  - [Evaluation](#evaluation)
- [Software 2: Alzheimer's Disease Detection](#Software-2-alzheimers-disease-detection)
  - [Datasets](#datasets-1)
  - [Data Preprocessing](#data-preprocessing-1)
  - [Training the Model](#training-the-model-1)
  - [Evaluation](#evaluation-1)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.8+
- GPU with CUDA support (recommended for training)
- [Conda](https://docs.conda.io/en/latest/miniconda.html) for environment management

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medical-image-analysis.git
   cd medical-image-analysis
2. Create a conda environment and install dependencies:
   ```bash
   conda create -n medimg python=3.8
   conda activate medimg
   pip install -r requirements.txt

## Software 1: nnUNetv2 for Biomedical Image Segmentation 

### Datasets

For this task, we will use several biomedical datasets such as BraTs2021, AMOS22, KiTS23, and BTCV. Download the datasets and organize them in the following structure:
 ```bash
    data/
        ├── BraTs2021/
        ├── AMOS22/
        ├── KiTS23/
        └── BTCV/
 ```
### Data Preprocessing

nnUNetv2 requires specific preprocessing steps. The preprocessing will include dataset analysis, resampling, normalization, data augmentation, and splitting.
1. Preprocess the raw data using the command -
  ```bash
   nnUNetv2_plan_and_preprocess -d <DATASET_ID>--verify_dataset_integrity
  ```

### Training the Model

1. To Train the preprocessed dataset
It is better to reduce the batch size to avoid the error "cuda: Out of memory"
There are 1,2,3,4,5 folds in training, where 0 is the least and so on.

-> For Training 2d data:
  ```bash
  nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD [--npz]
  ```
        
-> For Training 3d Data:
  ```bash
   nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD [--npz]
  ```

### Evaluation

1. Evaluate the nnUNetv2 Model
   ```bash
   python nnunetv2_evaluate.py --model_dir models --data_dir preprocessed_data
   ```

## Task 2: Alzheimer's Disease Detection

### Datasets
For the deep learning model training, T1-weighted structural MRI scans from the ADNI dataset are used. Over 3000 preprocessed scans are categorized into Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) based on memory tasks, adjusted for education level.
Download the ADNI dataset and organize it in the following structure:
```bash
data/
└── ADNI/
    ├── AD/
    ├── MCI/
    └── CN/
```

### Data Preprocessing

The preprocessing steps involve multiplanar reconstruction, Gradwarp correction, B1 non-uniformity correction, N3 intensity normalization, registration to standard space, segmentation, quality control, and data splitting.

1. Run the preprocessing script:
   ```bash
   python adni_preprocessing.py --data_dir data/ADNI --output_dir preprocessed_adni
   ```

### Training the Model

1. Train the deep learning model:
   ```bash
   python adni_train.py --data_dir preprocessed_adni --output_dir models/adni_model
   ```

### Evaluation

1. Evaluate the deep learning model:
   ```bash
   python adni_evaluate.py --model_dir models/adni_model --data_dir preprocessed_adni
   ```
   
