#!/bin/bash
#SBATCH --job-name=montonocity
#SBATCH --partition=gpu-test
#SBATCH --gres=gpu:A100:1
#SBATCH --time=3-10:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --output=./outputs/slurm-%j.log



export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python3 /gpfs/mariana/home/rkalak/xai_evaluation/project/Iot_Botnet_XAI/test_environment.py -f 'nboit_sample.csv' -d '/gpfs/mariana/home/rkalak/dataset/N-Balot/' -c 'class-1' -S 5000 -fn 100 -m 'accuracy' -fname 'fisher_score' -fcount 3 -ptype 'random_search' -rp '/gpfs/mariana/home/rkalak/xai_evaluation/project/Iot_Botnet_XAI/models/N-baIoT/fisherscore'

