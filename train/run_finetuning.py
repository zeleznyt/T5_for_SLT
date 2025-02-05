import os
import math
import json
import wandb
import torch
import evaluate
import numpy as np
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    T5Tokenizer, 
)
from model.configuration_t5 import SignT5Config
from model.modeling_t5 import T5ModelForSLT
from utils.translation import postprocess_text
from dataset.generic_sl_dataset import SignFeatureDataset as DatasetForSLT

from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
import yaml
import argparse

def init_wandb(config):
    wandb.login(
        key=os.getenv("WANDB_API_KEY")
    )
    wandb.init(
        project=config['TrainingArguments']["project_name"],
        tags=[config["ModelArguments"]["base_model_name"]],
        config=config,
    )
    wandb.run.name = '{}-{}'.format(wandb.run.name, config['TrainingArguments']['model_name'])

    return wandb


def set_seed(seed_value=42):
    # random.seed(seed_value)  # Python random
    np.random.seed(seed_value)  # NumPy random
    torch.manual_seed(seed_value)  # PyTorch (CPU & CUDA)
    torch.cuda.manual_seed(seed_value)  # GPU-specific seed
    torch.cuda.manual_seed_all(seed_value)  # Multi-GPU safe


def parse_args():
    """
    Parse command line arguments.
    All the arguments are set to None by default. Main source of arguments is the config file.
    Arguments set to a non-None value will override any arguments set by the config.
    Returns:
    args (argparse.Namespace): Parsed command line arguments
    """

    parser = argparse.ArgumentParser()

    # Configuration
    parser.add_argument("--config_file", type=str, default='config.yaml')

    # Core parameters
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--load_only_weights", type=bool, default=None)

    # Logging and saving
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--push_to_hub", type=bool, default=None)

    #  Debugging
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)

    # Training arguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lr_scheduler_type", type=str, default=None)
    parser.add_argument("--max_training_steps", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--fp16", type=bool, default=None)

    # Data processing
    parser.add_argument("--max_sequence_length", type=int, default=None)
    parser.add_argument("--max_token_length", type=int, default=None)
    parser.add_argument("--skip_frames", default=None)

    # Evaluation arguments
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--early_stopping", type=bool, default=None)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=None)

    # Other arguments
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def load_config(cfg_path):
    """
    Load config from a yaml file. 'none' and 'None' values are replaced by None value.
    Args:
        cfg_path: Path to config file
    Returns:
        config (dict): Config
    """
    with open(cfg_path, 'r') as file:
        cfg = yaml.safe_load(file)
    for param, value in cfg['TrainingArguments'].items():
        if value == 'none' or value == 'None':
            cfg['TrainingArguments'][param] = None

    # Add system job ID to config, if it exists
    system_cfg = ['PBS_JOBID', 'SLURM_JOB_ID']
    cfg['system'] = {}
    for variable in system_cfg:
        if variable in os.environ.keys():
            cfg['system'][variable] = os.environ[variable]
            cfg['TrainingArguments']['model_name'] = '{}-{}'.format(cfg['TrainingArguments']['model_name'], os.environ[variable])
    return cfg


def update_config(cfg, args):
    """
    Update config with args passed. Default None arguments are ignored.
    Args:
        cfg (dict): Config
        args (argparse.Namespace): Argument parsed from the command-line
    Returns:
        cfg (dict): Updated config
    """
    for k, v in vars(args).items():
        if k in cfg['TrainingArguments'] and v is not None:
            cfg['TrainingArguments'][k] = v
            if os.environ.get("LOCAL_RANK", "0") == "0" and args.verbose:
                print('Config value updated by args - {}: {}'.format(k, v))
    return cfg


if __name__ == "__main__":
    args = parse_args()
    if os.environ.get("LOCAL_RANK", "0") == "0" and args.verbose:
        print('Loading config...')
    config = load_config(args.config_file)
    config = update_config(config, args)

    training_config = config['TrainingArguments']
    model_config = config['ModelArguments']

    set_seed(training_config['seed'])

    if os.environ.get("LOCAL_RANK", "0") == "0" and training_config['report_to'] == 'wandb':
        init_wandb(config)
    else:
        os.environ["WANDB_DISABLED"] = "true"
    
    # Initialize the custom model
    t5_config = SignT5Config()
    for param, value in model_config.items():
        if param not in vars(t5_config):
            print('f{param} not in SignT5Config. It may be ignored...}')
        t5_config.__setattr__(param, value)

    if training_config['load_only_weights']:
        assert training_config['resume_from_checkpoint'], "resume_from_checkpoint must be provided when running with load_only_weights"
        model = T5ModelForSLT.from_pretrained(training_config['resume_from_checkpoint'], config=t5_config)
        training_config['resume_from_checkpoint'] = None
    else:
        model = T5ModelForSLT(config=t5_config)
    for param in model.parameters(): param.data = param.data.contiguous()
    tokenizer = T5Tokenizer.from_pretrained(model.config.base_model_name, clean_up_tokenization_spaces=True)

    if os.environ.get("LOCAL_RANK", "0") == "0" and training_config['report_to'] == 'wandb': # TODO: remove redundant data
        wandb.config.update(vars(model.config))

    # Add collate_fn to DataLoader
    def collate_fn(batch):
        # Add padding to the inputs 
        # "inputs" must be 250 frames long
        # "attention_mask" must be 250 frames long
        # "labels" must be 128 tokens long
        return {
            "sign_inputs": torch.stack([
                torch.cat((sample["sign_inputs"], torch.zeros(training_config['max_sequence_length'] - sample["sign_inputs"].shape[0], config['SignModelArguments']['projectors']['pose']['dim'])), dim=0)
                for sample in batch
            ]),
            "attention_mask": torch.stack([
                torch.cat((sample["attention_mask"], torch.zeros(training_config['max_sequence_length'] - sample["attention_mask"].shape[0])), dim=0)
                if sample["attention_mask"].shape[0] < training_config['max_sequence_length']
                else sample["attention_mask"]
                for sample in batch
            ]),
            "labels": torch.stack([
                torch.cat((sample["labels"].squeeze(0), torch.zeros(training_config['max_token_length'] - sample["labels"].shape[0])), dim=0)
                if sample["labels"].shape[0] < training_config['max_token_length']
                else sample["labels"]
                for sample in batch
            ]).squeeze(0).to(torch.long),
        }

    train_dataset = DatasetForSLT(tokenizer= tokenizer,
                                sign_data_args=config['SignDataArguments'],
                                split='train',
                                skip_frames=training_config['skip_frames'],
                                max_token_length=training_config['max_token_length'],
                                max_sequence_length=training_config['max_sequence_length'],
                                max_samples=training_config['max_train_samples'],
                                )

    val_dataset = DatasetForSLT(tokenizer= tokenizer,
                                sign_data_args=config['SignDataArguments'],
                                split='dev',
                                skip_frames=training_config['skip_frames'],
                                max_token_length=training_config['max_token_length'],
                                max_sequence_length=training_config['max_sequence_length'],
                                max_samples=training_config['max_val_samples'],
                                )

    if args.verbose:
        print(f"Training dataset: {len(train_dataset)}")
        print(f"Validation dataset: {len(val_dataset)}")

        # Print the first sample
        sample = train_dataset[0]

        print(f"Sign inputs:")
        print(sample["sign_inputs"])
        print(f"Attention mask:")
        print(sample["attention_mask"])
        print(f"Labels:")
        
    sacrebleu = evaluate.load('sacrebleu')

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        if isinstance(preds, tuple):
            preds = preds[0]
            preds = np.argmax(preds, axis=2)

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        if len(np.where(preds > len(tokenizer) - 1)[1]) > 0:
            print(f'Replacing <unk> for illegal tokens found on indexes {np.where(preds > len(tokenizer) - 1)[1]}')
        preds[preds > len(tokenizer) - 1] = tokenizer.unk_token_id
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        len_samples = 10 if len(decoded_preds) > 10 else len(decoded_preds)
        for i in range(len_samples):
            print(f"Prediction: {decoded_preds[i]}")
            print(f"Reference: {decoded_labels[i]}")
            print('*'*50)

        result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
        result = {
            "bleu": result["score"], 
            'bleu-1': result['precisions'][0],
            'bleu-2': result['precisions'][1],
            'bleu-3': result['precisions'][2],
            'bleu-4': result['precisions'][3],
        }

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}

        return result

    num_train_epochs = training_config['max_training_steps'] // (len(train_dataset) //
             training_config['per_device_train_batch_size'] // training_config['gradient_accumulation_steps'])
    num_train_epochs = max(math.ceil(num_train_epochs), 1)

    print(f"""
        Model: {training_config['model_name']}
        Training epochs: {num_train_epochs}
        Number of training steps: {training_config['max_training_steps']}
        Number of training batches: {len(train_dataset) // training_config['per_device_train_batch_size']}
        Number of validation examples: {len(val_dataset)}
    """)

    # Check if total batch size 128
    # assert args.per_device_train_batch_size * args.gradient_accumulation_steps == 128

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(training_config['output_dir'], training_config['model_name']),
        logging_steps=training_config['logging_steps'],
        num_train_epochs=num_train_epochs,
        # max_steps=args.max_training_steps,
        optim="adafactor",
        learning_rate=training_config['learning_rate'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        weight_decay=training_config['weight_decay'],
        per_device_train_batch_size=training_config['per_device_train_batch_size'],
        per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        eval_accumulation_steps=1,
        fp16=training_config['fp16'],
        push_to_hub=training_config['push_to_hub'],
        hub_model_id=training_config['model_name'],
        metric_for_best_model="bleu",
        save_total_limit=3,
        predict_with_generate=True,
        evaluation_strategy="steps",
        eval_steps=training_config['eval_steps'],
        save_strategy="steps",
        save_steps=training_config['save_steps'],
        generation_config=model.base_model.generation_config,
        ddp_find_unused_parameters=False,
    )

    if os.environ.get("LOCAL_RANK", "0") == "0" and training_config['report_to'] == 'wandb': # TODO: remove redundant data
        wandb.config.update(vars(training_args))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=training_config['resume_from_checkpoint'])

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_config['per_device_eval_batch_size'],
        collate_fn=collate_fn,
    )

    def evaluate_model(model, dataloader, tokenizer):

        predictions, labels = [], []
        for step, batch in tqdm(enumerate(dataloader)):
            batch = {k: v.to(model.base_model.device) for k, v in batch.items()}
            if len(batch['labels'].shape) < 2:
                batch['labels'] = batch['labels'].unsqueeze(0)
            outputs = model.generate(
                **batch,
                early_stopping=training_config['early_stopping'],
                no_repeat_ngram_size=training_config['no_repeat_ngram_size'],
                max_length=training_config['max_sequence_length'],
                num_beams=training_config['num_beams'],
                bos_token_id=tokenizer.pad_token_id,
            )

            # if len(np.where(outputs > len(tokenizer) - 1)[1]) > 0:
            if len(np.where(outputs.cpu().numpy() > len(tokenizer) - 1)[1]) > 0:
                print(f'Replacing <unk> for illegal tokens found on indexes {np.where(outputs.cpu().numpy() > len(tokenizer) - 1)[1]}')
            outputs[outputs > len(tokenizer) - 1] = tokenizer.unk_token_id

            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

            predictions.extend(decoded_preds)
            labels.extend([[translation] for translation in decoded_labels])

        return predictions, labels

    print('Evaluating model...')
    val_predictions, val_labels = evaluate_model(model, val_dataloader, tokenizer)

    # Save predictions to file
    with open(os.path.join(training_config['output_dir'], training_config['model_name'], "val_predictions.txt"), "w") as f:
        all_predictions = [
            {
                "prediction": prediction,
                "reference": label[0]
            }
            for prediction, label in zip(val_predictions, val_labels) 
        ]

        json.dump(all_predictions, f)
        print(f'Predictions saved to {os.path.join(training_config["output_dir"], training_config["model_name"], "val_predictions.txt")}')

    val_bleu = sacrebleu.compute(predictions=val_predictions, references=val_labels)

    # Save scores json
    scores = {
        "val": val_bleu,
    }

    with open(os.path.join(training_config["output_dir"], training_config["model_name"], "va_scores.json"), "w") as f:
        json.dump(scores, f)
        print(f'Scores saved to {os.path.join(training_config['output_dir'], training_config['model_name'], "val_scores.json")}')




