#!/bin/bash
#SBATCH --job-name=magv             # Name of your job
#SBATCH --output=%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_%j.err             # Error file
#SBATCH --partition=L40S              # Partition to submit to (A100, V100, etc.)


#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=30             # Request 8 CPU cores
#SBATCH --mem=32G                     # Request 32 GB of memory


# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Define variables for your job
# DATA_DIR="~/data"
# LR="1e-3"
# EPOCHS=100
# BATCH_SIZE=32

# Activate the environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate magv

# Execute the Python script with specific arguments
#srun python my_script.py --data $DATA_DIR --lr $LR --epochs $EPOCHS --batch-size $BATCH_SIZE
srun python train_refactor.py   --batch-size=16 \
                                    --cuda=1 \
                                    --dataset=/home/ids/flauron-23/fiftyone/open-images-v6 \
                                    --epochs=21 \
                                    --lambda=0.013 \
                                    --learning-rate=0.0001 \
                                    --model=cheng \
                                    --save=1 \
                                    --save-dir=../results/mask/adapt_0483 \
                                    --test-dir=/home/ids/flauron-23/kodak \
                                    --vanilla-adapt=1 \
                                    --num-workers=30 \
                                    --mask \
                                    --maxPrunning=0.4 \
                                    --nameRun=magv_40_cheng_14_pts_bis \
                                    --maxPoint=14 \
                                    --pruningType=unstructured
                                    
# Print job completion time
echo "Job finished at: $(date)"
