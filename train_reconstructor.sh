# Train despeckler
cleanup() {
    # Add your cleanup code here
    echo "Script interrupted, cleaning up..."
    # Kill any background processes
    jobs -p | xargs -r kill
    exit 1
}

# Set the trap
trap cleanup SIGINT SIGTERM

VERSION="your-choice"
DATABAND="your-choice"
DATADIR=/your/path/to/data
LOGDIR=training_logs # do not change this unless you know what you are doing
PATCH_SIZE=your-choice
STRIDE=your-choice
IN_CHANNELS=your-choice # 4 for 4 polar SAR images, 1 for single band
EPOCHS=your-choice
LATENT_SIZE=your-choice
LR=your-choice
MODEL="vae" # by defaut, it's a VAE

python scripts/train_reconstructor.py \
    --version $VERSION \
    --data_band $DATABAND \
    --datadir $DATADIR \
    --logdir $LOGDIR \
    --recon_model $MODEL \
    --recon_patch_size $PATCH_SIZE \
    --recon_stride $STRIDE \
    --recon_in_channels $IN_CHANNELS \
    --recon_epochs $EPOCHS \
    --recon_latent_size $LATENT_SIZE \
    --recon_lr_ae $LR

echo "Done dessous X!"
