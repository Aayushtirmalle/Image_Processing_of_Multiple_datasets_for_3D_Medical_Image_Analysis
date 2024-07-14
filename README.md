# Medical Image Analysis with nnUNetv2 and Alzheimer's Disease Detection

This repository provides the code and instructions to perform two key tasks in medical image analysis:
1. Biomedical image segmentation using nnUNetv2.
2. Alzheimer's disease detection using a deep learning model trained on structural MRIs.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Software 1: nnUNetv2 for Biomedical Image Segmentation](#Software-1-nnunetv2-for-biomedical-image-segmentation)
  - [Installation](#installation)
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

- For Software 1 
    - Python 3.8+
    - GPU with CUDA support (recommended for training)
    - Pytorch
    - [Conda](https://docs.conda.io/en/latest/miniconda.html) for environment management

- For Software 2
    - Python 
    - PyTorch
    - torchvision
    - progress
    - matplotlib
    - numpy
    - visdom
    - Clinica

## Software 1: nnUNetv2 for Biomedical Image Segmentation 

### Installation

1. Install the nnUNet software using interactive shell
   ```bash
        git clone https://github.com/MIC-DKFZ/nnUNet.git
        cd nnUNet
        pip install -e .
   ```


2. Create folders "nnUNet_raw, nnUNet_preprocessed, nnUNet_trained data" and set the path for the same, while using docker set the paths with respect to the docker path.

Locate the .bashrc file in your home folder of the docker and add the following lines to the bottom:
   ```bash
        export nnUNet_raw="/home/abc/nnUNetFrame/dataset/nnUNet_raw"
        export nnUNet_preprocessed="/home/abc/nnUNetFrame/dataset/nnUNet_preprocessed"
        export nnUNet_results="/home/abc/nnUNetFrame/dataset/nnUNet_results"
   ```
        
After editing the .bashrc folder if there is some error regarding the path, the "same lines" can be used temporarily(every time the docker is restarted it needs to be executed so prevent errors.

4. Install the hiddenlayer for nnUNet
  ```bash
        pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
  ```

### Datasets

For this task, we will use several biomedical datasets such as BraTs2021, AMOS22, KiTS23, and BTCV. Download the datasets and organize them in the following structure:
The Folders should be structured as follows 
  ```bash
    nnUNet_raw/
        ├── Dataset001_BraTs2021
        ├── Dataset002_AMOS22
        ├── Dataset003_KiTs23
        ├── Dataset004_BTCV
        ├── ...
```
Within wach dataset folder ,the following structure is expected:
```bash
Dataset001_BraTs2021/
├── dataset.json
├── imagesTr
├── imagesTs  # optional
└── labelsTr
```
- **imagesTr** contains the images belonging to the training cases. nnU-Net will perform pipeline configuration, training with cross-validation, as well as finding postprocessing and the best ensemble using this data.
- **imagesTs** (optional) contains the images that belong to the test cases. nnU-Net does not use them! This could just be a convenient location for you to store these images. Remnant of the Medical Segmentation Decathlon folder structure.
- **labelsTr** contains the images with the ground truth segmentation maps for the training cases.
- **dataset.json** contains metadata of the dataset.

For example for the first Dataset of the MSD: BraTs2021. This dataset hat four input channels: FLAIR (0000), T1w (0001), T1gd (0002) and T2w (0003). Note that the imagesTs folder is optional and does not have to be present.
```bash
nnUNet_raw/Dataset001_BraTs2021/
├── dataset.json
├── imagesTr
│   ├── BRATS_001_0000.nii.gz
│   ├── BRATS_001_0001.nii.gz
│   ├── BRATS_001_0002.nii.gz
│   ├── BRATS_001_0003.nii.gz
│   ├── BRATS_002_0000.nii.gz
│   ├── BRATS_002_0001.nii.gz
│   ├── BRATS_002_0002.nii.gz
│   ├── BRATS_002_0003.nii.gz
│   ├── ...
├── imagesTs
│   ├── BRATS_450_0000.nii.gz
│   ├── BRATS_450_0001.nii.gz
│   ├── BRATS_450_0002.nii.gz
│   ├── BRATS_450_0003.nii.gz
│   ├── BRATS_451_0000.nii.gz
│   ├── BRATS_451_0001.nii.gz
│   ├── BRATS_451_0002.nii.gz
│   ├── BRATS_451_0003.nii.gz
│   ├── ...
└── labelsTr
    ├── BRATS_001.nii.gz
    ├── BRATS_002.nii.gz
    ├── ...

```


### Data Preprocessing

nnUNetv2 requires specific preprocessing steps. The preprocessing will include dataset analysis, resampling, normalization, data augmentation, and splitting.
1. Preprocess the raw data using the command -
  ```bash
   nnUNetv2_plan_and_preprocess -d <DATASET_ID>--verify_dataset_integrity
  ```

### Training the Model

- To Train the preprocessed dataset
It is better to reduce the batch size to avoid the error "cuda: Out of memory"
There are 5 folds in training that is [0,1,2,3,4], where 0 is the least and so on.

-> For Training 2d data:
  ```bash
  nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD [--npz]
  ```
        
-> For Training 3d Data:
  ```bash
   nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres FOLD [--npz]
  ```
The trained models will be written to the nnUNet_results folder. Each training obtains an automatically generated output folder name:

nnUNet_results/DatasetXXX_MYNAME/TRAINER_CLASS_NAME__PLANS_NAME__CONFIGURATION/FOLD

For Dataset002_AMOS22 (from the MSD), for example, this looks like this:
```bash
nnUNet_results/
├── Dataset002_AMOS22
    └── nnUNetTrainer__nnUNetPlans__3d_fullres
         ├── fold_0
         ├── fold_1
         ├── fold_2
         ├── fold_3
         ├── fold_4
         ├── dataset.json
         ├── dataset_fingerprint.json
         └── plans.json
```
### Evaluation

1. Evaluate the nnUNetv2 Model
   ```bash
   python nnunetv2_evaluate.py --model_dir models --data_dir preprocessed_data
   ```

## Software 2: Alzheimer's Disease Detection

### Datasets
For the deep learning model training, T1-weighted structural MRI scans from the ADNI dataset are used. Over 3000 preprocessed scans are categorized into Alzheimer's Disease (AD), Mild Cognitive Impairment (MCI), and Cognitively Normal (CN) based on memory tasks, adjusted for education level.

- Downloading the ADNI dataset:
    - Request approval and register at [ADNI Website](https://adni.loni.usc.edu/data-samples/access-data/)
    - Download the scans. From the main page click on PROJECTS and ADNI. Then click on Download and choose Image collections. In the Advanced search tab, untick ADNI 3 and tick MRI to download all the MR images.
    - In the Advanced search results tab, click Select All and Add To Collection. Finally, in the Data Collection tab, select the collection you just created, tick All and click on Advanced download. We advise you to group files as 10 zip files


### Data Preprocessing

- Convert data to BIDS format:
  We'll be using clinica software to do this. Note that we first preprocess the training set to generate the template and use the template to preprocess validation and test set. Template is provided which is used to do the conversion.
```bash
run_convert.sh
```

- Preprocess converted and splitted data:
  To do this task the following script can be used.
  ```bash
  run_adni_preprocess.sh
  ```

- For Validation and Test:
  To do this task the following script can be used.
  ```bash
  run_adni_preprocess_val.sh
  ```
  ```bash
  run_adni_preprocess_test.sh
  ```



### Training the Model

- To Train the deep learning model:
  Use the Python Script to train
  ```bash
  python main.py
  ```
  Own config files can be created. Add --config flag to the config file


### Evaluation

1. Evaluate the deep learning model:
   To do the evaluation, I have used this "eval.ipynb" file

![Description of Image]("C:\Users\User\Pictures\Picture1.png")


   
