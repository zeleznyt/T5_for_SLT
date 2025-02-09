echo "Running predict script.."
python eval/predict.py \
	--config_file configs/predict_H2S_t5-base.yaml \
	--model_dir results/5-1-YT-ASL-split_constant0.001/_checkpoint-12800 \
	--output_dir results/5-1-YT-ASL-split_constant0.001/H2S_predict-12800 \
	--batch_size 8 \
	--verbose
