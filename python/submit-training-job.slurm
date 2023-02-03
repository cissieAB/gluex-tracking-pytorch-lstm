#!/bin/bash

#SBATCH --gres=gpu:T4:1
#SBATCH -N 1  # number of nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000  # MB
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --output=jlog_%j.out
#SBATCH --job-name=lstm-train

set -euxo pipefail

srun nvidia-smi

env

CONDA_DIR=/apps/anaconda3/2021.05/etc/profile.d  # depend on the farm config
WK_DIR=/home/xmei/projects/gluex-tracking-pytorch-lstm/python

source /etc/profile.d/modules.sh
module use /apps/modulefiles
module load anaconda3
which conda
conda --version

sh $CONDA_DIR/conda.sh
conda-env list
pwd
cd $WK_DIR
pwd
# A100 GPU requires higher CUDA version than the ifarm default
# This env is tested in Nov, 2022 and was working
source activate pytorch-cuda11_6  # "source activate" instead of "conda activate"

srun python LSTM_training.py

srun python validation_processing.py

mkdir job_${SLURM_JOBID}
mv *.png *.pt *.log *.out job_${SLURM_JOBID}

