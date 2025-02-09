export WANDB_ENTITY="jsalt2024-slt"
export WANDB_PROJECT="T5_SSL"
export WANDB_API_KEY="your-wandb-api-key" # optional, can be pass as an argument as well

export HF_TOKEN="your-hf-token"

echo "Running training script.."
python train/run_finetuning.py \
    --config_file configs/mT5-base_SSL-A.yaml \
    --verbose \
    --wandb_api_key key_or_path_to_your_key # optional. only if exporting the variable is not possible due to security reasons