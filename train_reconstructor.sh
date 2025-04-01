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

# VERSION="0"
# DATABAND="L"
# CKPTDIR=$RECONSTRUCTOR_WORKDIR/weights_storage/version_$VERSION/despeckler/
# DATADIR=$WORKDIR/projects/data/


# srun --kill-on-bad-exit=1 python train_despeckler.py \
#                             --version $VERSION \
#                             --data_band $DATABAND \
#                             --despeckler_ckpt_path $CKPTDIR\
#                             --despeckler_predict_datadir $DATADIR \

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

VERSION="Strasbourg"
DATABAND="X"
DATADIR=/mnt/DATA/data/AnomalyDetection/Strasbourg/short/Traites_EOST_npy
LOGDIR=training_logs
PATCH_SIZE=256
STRIDE=100
IN_CHANNELS=1
EPOCHS=500
LATENT_SIZE=256
LR=5e-4

# python scripts/train_reconstructor.py \
#     --version $VERSION \
#     --data_band $DATABAND \
#     --datadir $DATADIR \
#     --logdir $LOGDIR \
#     --recon_patch_size $PATCH_SIZE \
#     --recon_stride $STRIDE \
#     --recon_in_channels $IN_CHANNELS \
#     --recon_epochs $EPOCHS \
#     --recon_latent_size $LATENT_SIZE \
#     --recon_lr_ae $LR

# echo "Done Strasbourg!"

VERSION="dessous_AAE"
DATADIR=/mnt/DATA/data/AnomalyDetection/dessous
STRIDE=16
LATENT_SIZE=128
IN_CHANNELS=4
EPOCHS=100
LR=1e-3
PATCH_SIZE=64

python scripts/train_reconstructor.py \
    --version $VERSION \
    --data_band $DATABAND \
    --datadir $DATADIR \
    --logdir $LOGDIR \
    --recon_patch_size $PATCH_SIZE \
    --recon_stride $STRIDE \
    --recon_in_channels $IN_CHANNELS \
    --recon_epochs $EPOCHS \
    --recon_latent_size $LATENT_SIZE \
    --recon_lr_ae $LR

echo "Done dessous X!"

# VERSION="dessous_L"
# DATABAND="L"
# STRIDE=32

# python scripts/train_reconstructor.py \
#     --version $VERSION \
#     --data_band $DATABAND \
#     --datadir $DATADIR \
#     --logdir $LOGDIR \
#     --recon_patch_size $PATCH_SIZE \
#     --recon_stride $STRIDE \
#     --recon_in_channels $IN_CHANNELS \
#     --recon_epochs $EPOCHS \
#     --recon_latent_size $LATENT_SIZE \
#     --recon_lr_ae $LR

# echo "Done dessous L!"

# VERSION="Nimes"
# DATABAND="X"
# STRIDE=64
# DATADIR=/mnt/DATA/data/AnomalyDetection/Nimes

# python scripts/train_reconstructor.py \
#     --version $VERSION \
#     --data_band $DATABAND \
#     --datadir $DATADIR \
#     --logdir $LOGDIR \
#     --recon_patch_size $PATCH_SIZE \
#     --recon_stride $STRIDE \
#     --recon_in_channels $IN_CHANNELS \
#     --recon_epochs $EPOCHS \
#     --recon_latent_size $LATENT_SIZE \
#     --recon_lr_ae $LR

# echo "Done Nimes X!"
