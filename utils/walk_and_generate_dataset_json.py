import sys
sys.path.append('/vol/csedu-nobackup/course/IMC037_aimi/group18/jakob/nnUNet/nnunetv2/dataset_conversion')

from generate_dataset_json import generate_dataset_json

import os
import json
import argparse

def count_files(directory):
    """Recursively counts files in a directory."""
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

def walk_and_generate_dataset_json(base_directory):
    """Walk through every subdirectory in base_directory and generate dataset.json."""
    
    for root, dirs, files in os.walk(base_directory):
        # Skip the root directory itself
        if root == base_directory:
            continue
        
        # Check if both 'imagesTr' and 'labelsTr' exist in this directory
        images_tr_dir = os.path.join(root, "imagesTr")
        labels_tr_dir = os.path.join(root, "labelsTr")
        
        if not os.path.exists(images_tr_dir) or not os.path.exists(labels_tr_dir):
            continue  # Skip this directory if it doesn't have both 'imagesTr' and 'labelsTr'

        
        channel_names = '{"0": "CT"}'
        labels = '{"background": 0, "tumor": 1}'
        
        # Correct path to count files in imagesTr
        num_training_cases = count_files(images_tr_dir)
        
        file_ending = ".nii.gz"
        
        # Generate the dataset.json for this dataset directory
        output_folder = root
        os.makedirs(output_folder, exist_ok=True)
        
        # Call the function to generate dataset.json
        generate_dataset_json(
            output_folder=output_folder,
            channel_names=json.loads(channel_names),
            labels=json.loads(labels),
            num_training_cases=num_training_cases,
            file_ending=file_ending,
            license="AIMI",
            converted_by="Jakob"
        )
        print(f"Generated dataset.json for {root}")

def main():
    parser = argparse.ArgumentParser(description="Walk through directories and generate dataset.json for each.")
    parser.add_argument('--base_directory', type=str, required=True, help="Base directory containing all datasets.")
    
    args = parser.parse_args()
    
    walk_and_generate_dataset_json(args.base_directory)

if __name__ == "__main__":
    main()
