#!/bin/bash -e
#SBATCH --partition=csedu
#SBATCH --account=cseduimc037
#SBATCH --gres=gpu:1                   
#SBATCH --cpus-per-task=4             
#SBATCH --mem=16G                      
#SBATCH --time=6:00:00                
#SBATCH --output=tmp/nnunet_train_%j.out
#SBATCH --error=tmp/nnunet_train_%j.err
#SBATCH --mail-user=daniel.duhnev@ru.nl
#SBATCH --mail-type=BEGIN,END,FAIL

# 1. Activate nnU-Net environment
source /vol/csedu-nobackup/course/IMC037_aimi/group18/daniel/venv/bin/activate

# 2. Ensure nnUNet CLI is on PATH
export PATH="/vol/csedu-nobackup/course/IMC037_aimi/group18/daniel/venv/bin:$PATH"

# 3. (Optional) Triton cache
export TRITON_CACHE_DIR="/scratch/$USER/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

# avoid matplotlib errors
export MPLCONFIGDIR="/scratch/test/.config/matplotlib"
mkdir -p "$MPLCONFIGDIR"

# 4. nnU-Net directories
export nnUNet_raw="/vol/csedu-nobackup/course/IMC037_aimi/group18/daniel/nnUNet_raw"
export nnUNet_preprocessed="/vol/csedu-nobackup/course/IMC037_aimi/group18/daniel/nnUNet_preprocessed"
export nnUNet_results="/vol/csedu-nobackup/course/IMC037_aimi/group18/daniel/nnUNet_results"

# Train using your custom trainer of choice
nnUNetv2_train \
   -tr CustomIntensityOnlyTrainer \
  1 3d_fullres 0
