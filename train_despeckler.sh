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
EPOCHS="your-choice"
STRIDE="your-choice"
LOGDIR=training_logs # do not change this unless you know what you are doing
DATADIR=/your/path/to/data

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
