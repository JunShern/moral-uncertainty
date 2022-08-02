#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=tune_ethics
#SBATCH --output=logs/slurm_%A.out

set -x
NGPUS=$1
MODEL=$2
LR=$3
BS=$4
EPOCH=$5
GA=$6

python tune.py --ngpus $NGPUS --model $MODEL --learning_rate $LR --batch_size $BS --nepochs $EPOCH --gradient_acc_steps $GA --verbose