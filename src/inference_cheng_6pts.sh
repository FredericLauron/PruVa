#!/bin/bash
#SBATCH --job-name=magv             # Name of your job
#SBATCH --output=%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=%x_%j.err             # Error file
#SBATCH --partition=mm              # Partition to submit to (A100, V100, etc.)


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
srun python inference_cheng_6pts.py   
                                    
# Print job completion time
echo "Job finished at: $(date)"
