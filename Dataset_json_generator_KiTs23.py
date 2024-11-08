from batchgenerators.utilities.file_and_folder_operations import *
import shutil
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import os
import re

def convert_kits2023(kits_base_dir: str, nnunet_dataset_id: int = 220):
    task_name = "KiTS2023"
    foldername = "Dataset%03.0d_%s" % (nnunet_dataset_id, task_name)

    # Set up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    # Define regex pattern to match kidney and tumor instance files
    pattern = re.compile(r'^(kidney|tumor)_instance-(\d+)_annotation-(\d+)\.nii\.gz$')
    cases = subdirs(kits_base_dir, prefix='case_', join=False)

    for case in cases:
        case_path = join(kits_base_dir, case, 'instances')
        image_files = []
        label_files = []

        for file in os.listdir(case_path):
            match = pattern.match(file)
            if match:
                organ = match.group(1)
                instance = match.group(2)
                annotation = match.group(3)

                # Rename and organize images and labels
                if organ == "kidney":
                    image_dest = join(imagestr, f'{case}_0000.nii.gz')  # Single image file per case
                    shutil.copy(join(case_path, file), image_dest)
                    image_files.append(image_dest)
                elif organ == "tumor":
                    label_dest = join(labelstr, f'{case}.nii.gz')  # Single label file per case
                    shutil.copy(join(case_path, file), label_dest)
                    label_files.append(label_dest)

    # Generate the dataset JSON file
    generate_dataset_json(out_base, {0: "CT"},
                          labels={
                              "background": 0,
                              "kidney": 1,
                              "tumor": 2
                          },
                          regions_class_order=(1, 2),
                          num_training_cases=len(cases), 
                          file_ending='.nii.gz',
                          dataset_name=task_name, 
                          reference='none',
                          release='0.1.3',
                          overwrite_image_reader_writer='NibabelIOWithReorient',
                          description="KiTS2023 dataset")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_folder', type=str, help="The KiTS2023 dataset base directory with structured instances")
    parser.add_argument('-d', required=False, type=int, default=220, help='nnU-Net Dataset ID, default: 220')
    args = parser.parse_args()
    kits_base = args.input_folder
    convert_kits2023(kits_base, args.d)
