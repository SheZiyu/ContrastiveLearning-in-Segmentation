#!/bin/bash
#SBATCH --partition=gpulong
#SBATCH --container-image=/home/zshe/zshe+deep_learning+v4.sqsh
#SBATCH --job-name=gcl
#SBATCH --no-container-entrypoint
#SBATCH --container-mounts=/home/zshe/practice:/practice
#SBATCH --gres=gpu:20gb:1
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=4
#SBATCH --time=71:59:00


cd /practice/AMOS22/AMOS22
/usr/bin/python3 /practice/AMOS22/AMOS22/train_con_our.py


 