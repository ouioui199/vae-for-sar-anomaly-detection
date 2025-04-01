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

VERSION="your_version_here"
DATABAND="your_data_band_here"
EPOCHS="desired_epochs_here"
STRIDE="desired_stride_here"
LOGDIR=training_logs
DATADIR=/your/data/directory/data_folder1/

python scripts/train_despeckler.py \
    --version $VERSION \
    --despeckler_epochs $EPOCHS \
    --logdir $LOGDIR \
    --datadir $DATADIR \
    --data_band $DATABAND \
    --despeckler_pol_channels Hh \
    --despeckler_stride $STRIDE

echo "Finished training despeckling on Hh polarization"

python scripts/train_despeckler.py \
    --version $VERSION \
    --despeckler_epochs $EPOCHS \
    --logdir $LOGDIR \
    --datadir $DATADIR \
    --data_band $DATABAND \
    --despeckler_pol_channels Hv \
    --despeckler_stride $STRIDE

echo "Finished training despeckling on Hv polarization"

python scripts/train_despeckler.py \
    --version $VERSION \
    --despeckler_epochs $EPOCHS \
    --logdir $LOGDIR \
    --datadir $DATADIR \
    --data_band $DATABAND \
    --despeckler_pol_channels Vh \
    --despeckler_stride $STRIDE

echo "Finished training despeckling on Vh polarization"

python scripts/train_despeckler.py \
    --version $VERSION \
    --despeckler_epochs $EPOCHS \
    --logdir $LOGDIR \
    --datadir $DATADIR \
    --data_band $DATABAND \
    --despeckler_pol_channels Vv \
    --despeckler_stride $STRIDE

echo "Finished training despeckling on Vv polarization"
