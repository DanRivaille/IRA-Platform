#!/bin/bash
#SBATCH --job-name=TrnTstML
#SBATCH --output=/scratch/ivan.santos/test_models/output/train_and_test.out
#SBATCH --error=/scratch/ivan.santos/test_models/error/train_and_test.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=02:00:00
#SBATCH --partition GPU

ROOT_PROJECT_FOLDER=/home/ivan.santos/repositories/IRA-Platform
SOURCE_FOLDER=$ROOT_PROJECT_FOLDER/src

#source $HOME/miniconda3/etc/profile.d/conda.sh
#conda activate IRA

#python3 $ROOT_PROJECT_FOLDER/main.py --id test_ae --config $ROOT_PROJECT_FOLDER/config_files/config_full_ae2.json --save
python3 $ROOT_PROJECT_FOLDER/main.py --id test_ae --config $ROOT_PROJECT_FOLDER/config_files/config_example.json --save
#python3 $ROOT_PROJECT_FOLDER/main.py --id test_ae --config $ROOT_PROJECT_FOLDER/config_files/config_full_ae2.json --save --test
python3 $ROOT_PROJECT_FOLDER/main.py --id test_ae --config $ROOT_PROJECT_FOLDER/config_files/config_example.json --save --test
