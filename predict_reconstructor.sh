#!/bin/bash                      
#SBATCH --time=00:30:00
#SBATCH --qos=co_short_gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=g317,g318,g319
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --job-name=pred_despeckler
#SBATCH --output=pred_despeckler.txt

# # Train despeckler slurm script
# export RECONSTRUCTOR_WORKDIR=$WORKDIR/projects/reconstructor

# if [[ ! -d "$RECONSTRUCTOR_WORKDIR" ]]; then
#     mkdir $RECONSTRUCTOR_WORKDIR
# fi

# VERSION="dessous"
# DATABAND="X"
# DATADIR=/mnt/DATA/data/AnomalyDetection/dessous/

# srun --kill-on-bad-exit=1 python scripts/predict_reconstructor.py \
#                             --version $VERSION \
#                             --data_band $DATABAND \
#                             --datadir $DATADIR \

# echo "Done!"

# Train despeckler ubuntu pc script
cleanup() {
    # Add your cleanup code here
    echo "Script interrupted, cleaning up..."
    # Kill any background processes
    jobs -p | xargs -r kill
    exit 1
}

# Set the trap
trap cleanup SIGINT SIGTERM

VERSION="dessous_VAE_L2"
DATABAND="X"
DATADIR=/mnt/DATA/data/AnomalyDetection/dessous/
LATENT_SIZE=256
# LATENT_SIZE=128
PATCH_SIZE=256
# PATCH_SIZE=64
ANOMALY_KERNEL=11
STRIDE=1

python scripts/predict_reconstructor.py \
    --version $VERSION \
    --data_band $DATABAND \
    --datadir $DATADIR \
    --recon_latent_size $LATENT_SIZE \
    --recon_patch_size $PATCH_SIZE \
    --recon_anomaly_kernel $ANOMALY_KERNEL \
    --recon_stride $STRIDE \
    --recon_sample_prediction

echo "Done!"
