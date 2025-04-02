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
LATENT_SIZE=your-choice
PATCH_SIZE=your-choice
ANOMALY_KERNEL=your-choice
STRIDE=1 # Stride = 1 for no overlaping in predicted image and best image quality.

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
