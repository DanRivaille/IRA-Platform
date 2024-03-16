#!/bin/bash
#SBATCH --job-name=test_server
#SBATCH --output=/scratch/ivan.santos/test.out
#SBATCH --error=/scratch/ivan.santos/test.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=00:15:00
#SBATCH --partition GPU

### to verify your gpu is cuda enable check
#lspci | grep -i nvidia

### gcc compiler is required for development using the cuda toolkit. to verify the version of gcc install enter
#gcc --version

# Finally, to verify the installation, check
nvidia-smi
nvcc -V

source /home/ivan.santos/miniconda3/etc/profile.d/conda.sh

conda activate myenv-cuda

### to verify your gpu is cuda enable check
python3 scripts/python_scripts/keras_test.py

conda deactivate
