T5X_MODEL_CHECKPOINT="gs://t5-data/pretrained_models/t5x/t5_base/checkpoint_999900"
CONFIG_PATH="https://huggingface.co/google-t5/t5-base/resolve/main/config.json"

DOWNLOAD_DIR="t5x-base/"
mkdir -p $DOWNLOAD_DIR
gsutil -m cp -r $T5X_MODEL_CHECKPOINT $DOWNLOAD_DIR
cd $DOWNLOAD_DIR
wget $CONFIG_PATH
cd ..