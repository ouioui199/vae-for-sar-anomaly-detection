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

python scripts/predict_despeckler.py \
    --version $VERSION \
    --data_band $DATABAND \
    --datadir $DATADIR \

echo "Done!"
