#!/bin/bash
#SBATCH --job-name=runParseFiles
#SBATCH --output=/scratch/ivan.santos/z24/output/parse_files.out
#SBATCH --error=/scratch/ivan.santos/z24/error/parse_files.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=00:45:00
#SBATCH --partition LM

ROOT_PROJECT_FOLDER=/home/ivan.santos/repositories/IRA-Platform
SOURCE_FOLDER=$ROOT_PROJECT_FOLDER/src

python3 $ROOT_PROJECT_FOLDER/scripts/python_scripts/parseZipFiles.py
