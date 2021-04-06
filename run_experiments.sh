#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=10GB
#SBATCH --output=comsoc.out
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=d.j.krol.1@student.rug.nl

# Get the resources we need
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
pip install pycuda --user
pip install scikit-learn --user

#Run the training
python main.py -n 20 -e 5
