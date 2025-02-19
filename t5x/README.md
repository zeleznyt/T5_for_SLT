# T5X
## Guide to get T5X pretrained checkpoints
### Prerequisites
Based on [official transformers utility guide](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/convert_t5x_checkpoint_to_pytorch.py)
1. Install [T5X](https://github.com/google-research/t5x)
```
git clone --branch=main https://github.com/google-research/t5x
cd t5x

python3 -m pip install -e '.[tpu]' -f \
  https://storage.googleapis.com/jax-releases/libtpu_releases.html

cd ..
```
2. Install gsutil according to https://cloud.google.com/storage/docs/gsutil_install

### Download original T5X checkpoint
```
# Example
bash t5x_get_t5-base.sh
```
<details>
  <summary>Full details</summary>

* Get a T5X checkpoint at https://github.com/google-research/t5x/blob/main/docs/models.md#t5-11-checkpoints 
```
# Example:
gsutil -m cp -r gs://t5-data/pretrained_models/t5x/t5_1_1_small $HOME/
```
* Create or download a corresponding config for the downloaded model. E.g. for T5 v1.1 small, you can use
    https://huggingface.co/google/t5-v1_1-small/blob/main/config.json
</details>


### Convert to pytorch
```
# Example
MODEL_PATH="t5x-base"
python convert_t5x_checkpoint_to_pytorch.py \
    --t5x_checkpoint_path=$MODEL_PATH/checkpoint_999900 \
    --config_file=$MODEL_PATH/config.json \
    --pytorch_dump_path=$MODEL_PATH/t5x-base-pytorch
```
