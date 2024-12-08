# Image Processing of Multiple Datasets for 3D Medical Image Analysis

This repository contains the code and resources for the project "Image Processing of Multiple Datasets for 3D Medical Image Analysis", which focuses on training and evaluating models for medical image segmentation and Alzheimer's disease detection using deep learning techniques.

### Project Overview

This project utilized two state-of-the-art frameworks for medical image analysis:
1.  nnUNetv2 : A self-configuring deep learning framework designed for biomedical image segmentation.
2.  CNN Design for Alzheimer's Disease (AD):A convolutional neural network (CNN) tailored for early detection of Alzheimer's disease using MRI data. 

The primary objective was to preprocess, train, and evaluate these models using public datasets, analyzing their performance on segmentation and classification tasks.

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

For example for the first Dataset of the MSD: BraTs2021. This dataset has four input channels: FLAIR (0000), T1w (0001), T1gd (0002) and T2w (0003). Note that the imagesTs folder is optional and does not have to be present.
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
   Used this "eval.ipynb" file for evaluating.


### Reference 

- [Github repository for nnUNetv2](https://github.com/MIC-DKFZ/nnUNet)
- [Github repository for CNN_design_for_AD](https://github.com/NYUMedML/CNN_design_for_AD)


   
