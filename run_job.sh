#!/bin/bash
#SBATCH --job-name=first_test_seq2seq2
#SBATCH --account=fc_horowitz
#SBATCH --partition=savio2_1080ti
#SBATCH --time=00:10:00
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=simon.ehlers@berkeley.edu

## Command(s) to run:
module load python
module load tensorflow/1.10.0-py36-pip-gpu
module load libpng gcc gdal proj glibc

python attention_encoder_decoder_test.py > job.pyout
