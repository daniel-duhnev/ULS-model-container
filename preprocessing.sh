#!/bin/bash -e
#SBATCH --partition=csedu
#SBATCH --account=cseduimc037
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=tmp/nnunet_preprocess-%j.out
#SBATCH --error=tmp/nnunet_preprocess-%j.err
#SBATCH --mail-user=daniel.duhnev@ru.nl
#SBATCH --mail-type=BEGIN,END,FAIL

# Activate your nnU-Net virtual environment
source /vol/csedu-nobackup/course/IMC037_aimi/group18/daniel/venv/bin/activate

# Prepare logging dir
mkdir -p tmp

# Ensure nnU-Net CLI is on your PATH
export PATH="/vol/csedu-nobackup/course/IMC037_aimi/group18/daniel/venv/bin:$PATH"

# Optional: Triton cache
export TRITON_CACHE_DIR="/vol/csedu-nobackup/course/IMC037_aimi/group18/daniel/tmp/triton_cache"
mkdir -p $TRITON_CACHE_DIR

# nnU-Net directory variables
export nnUNet_raw="/vol/csedu-nobackup/course/IMC037_aimi/group18/daniel/nnUNet_raw"
export nnUNet_preprocessed="/vol/csedu-nobackup/course/IMC037_aimi/group18/daniel/nnUNet_preprocessed"
export nnUNet_results="/vol/csedu-nobackup/course/IMC037_aimi/group18/daniel/nnUNet_results"

# Prepare (only if dataset.json or raw data changed)
nnUNetv2_plan_and_preprocess -d 1
