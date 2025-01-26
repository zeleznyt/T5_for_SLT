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

def init_wandb(args):

    if args.report_to != 'wandb':
        os.environ["WANDB_DISABLED"] = "true"
        print("Disabling wandb. To enable, set report_to: 'wandb'")
        return

    if args.dev:
        os.environ["WANDB_DISABLED"] = "true"
        print("Running in dev mode, disabling wandb")
        return

    wandb.login(
        key=os.getenv("WANDB_API_KEY")
    )

    system_config = ['PBS_JOBID', 'SLURM_JOB_ID']
    config= {'args': vars(args), 'system': {}}
    for variable in system_config:
        if variable in os.environ.keys():
            config['system'][variable] = os.environ[variable]

    wandb.init(
        project=args.project_name,
        # name=args.model_name,
        # tags=[args.dataset_type, args.transform, args.modality] + (["dev"] if args.dev else []) + (["sweep"] if args.sweep else []),
        config=config,
    )

    return wandb

def parse_args():

    import argparse

    parser = argparse.ArgumentParser()

    # Configuration
    parser.add_argument("--config_file", type=str, default=None)

    # Required parameters
    parser.add_argument("--model_name", type=str, default="T5_for_SLT")
    # parser.add_argument("--dataset_type", type=str, default="how2sign", choices=["how2sign", "yasl"])
    # parser.add_argument("--dataset_dir", type=str, default='/home/kara-nlp/Documents/Repositories/Thesis/SLT/Datasets/How2Sign/Mediapipe')
    parser.add_argument("--output_dir", default='./results',type=str)
    parser.add_argument("--seed", type=int, default=42)
    
    # # New data scheme
    # parser.add_argument('--annotation_file', type=str)
    # parser.add_argument('--metadata_file', type=str)

    # Data processing
    parser.add_argument("--skip_frames", default=None)
    parser.add_argument("--max_token_length", type=int, default=128)
    parser.add_argument("--max_sequence_length", type=int, default=250)
    parser.add_argument("--transform", type=str, default="yasl", choices=["yasl", "custom"])
    # parser.add_argument("--modality", type=str, default="pose", choices=["pose", "sign2vec", "mae"])

    # Training arguments
    # parser.add_argument("--embedding_dim", type=int, default=255)
    parser.add_argument("--model_id", type=str, default="t5-small")
    parser.add_argument("--max_training_steps", type=int, default=20_000) # TODO
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=float, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--lr_scheduler_type", type=str, default='constant')
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--report_to", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--pose_dim", type=int, default=208)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.0)

    # Evaluation arguments
    parser.add_argument("--num_beams", type=int, default=5)
    # parser.add_argument("--length_penalty", type=float, default=0.6)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)

    # Running arguments
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--project_name", type=str, default="h2s-t5")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    # parser.add_argument("--is_normalized", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--load_only_weights", action="store_true")

    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def load_config(args):

    import yaml

    if args.config_file is None:
        raise ValueError("Please provide a config_file")

    if args.config_file is not None:
        with open(args.config_file, "r") as f:
            config = yaml.safe_load(f)

        # Update the default arguments with the config file
        for key, value in config.items():
            setattr(args, key, value)

    if args.model_name and 'PBS_JOBID' in os.environ.keys():
        args.model_name = args.model_name + '_' + os.environ['PBS_JOBID']
    if args.model_name and 'SLURM_JOB_ID' in os.environ.keys():
        args.model_name = args.model_name + '_' + os.environ['SLURM_JOB_ID']

    return args

if __name__ == "__main__":

    args = parse_args()
    args = load_config(args)

    if os.environ.get("LOCAL_RANK", "0") == "0":
        init_wandb(args)
    
    # Initialize the custom model
    model_config = SignT5Config()
    for param, value in args.ModelArguments.items():
        if param not in vars(model_config):
            print('f{param} not in SignT5Config. It may be ignored...}')
        model_config.__setattr__(param, value)

    if args.load_only_weights:
        assert args.resume_from_checkpoint, "resume_from_checkpoint must be provided when running with load_only_weights"
        model = T5ModelForSLT.from_pretrained(args.resume_from_checkpoint, config=model_config)
        args.resume_from_checkpoint = None
    else:
        model = T5ModelForSLT(config=model_config)
    for param in model.parameters(): param.data = param.data.contiguous()
    tokenizer = T5Tokenizer.from_pretrained(args.model_id)

    if args.report_to == 'wandb':
        wandb.config.update(vars(model.config))

    # Add collate_fn to DataLoader
    def collate_fn(batch):
        # Add padding to the inputs 
        # "inputs" must be 250 frames long
        # "attention_mask" must be 250 frames long
        # "labels" must be 128 tokens long
        return {
            "sign_inputs": torch.stack([
                torch.cat((sample["sign_inputs"], torch.zeros(args.max_sequence_length - sample["sign_inputs"].shape[0], args.pose_dim)), dim=0)
                for sample in batch
            ]),
            "attention_mask": torch.stack([
                torch.cat((sample["attention_mask"], torch.zeros(args.max_sequence_length - sample["attention_mask"].shape[0])), dim=0)
                if sample["attention_mask"].shape[0] < args.max_sequence_length
                else sample["attention_mask"]
                for sample in batch
            ]),
            "labels": torch.stack([
                torch.cat((sample["labels"].squeeze(0), torch.zeros(args.max_token_length - sample["labels"].shape[0])), dim=0)
                if sample["labels"].shape[0] < args.max_token_length
                else sample["labels"]
                for sample in batch
            ]).squeeze(0).to(torch.long),
        }

    train_dataset = DatasetForSLT(tokenizer= tokenizer,
                                sign_data_args=args.SignDataArguments,
                                split='train',
                                skip_frames=args.skip_frames,
                                max_token_length=args.max_token_length,
                                max_sequence_length=args.max_sequence_length,
                                max_samples=args.max_train_samples,
                                )

    val_dataset = DatasetForSLT(tokenizer= tokenizer,
                                sign_data_args=args.SignDataArguments,
                                split='dev',
                                skip_frames=args.skip_frames,
                                max_token_length=args.max_token_length,
                                max_sequence_length=args.max_sequence_length,
                                max_samples=args.max_val_samples,
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

    num_train_epochs = args.max_training_steps // (len(train_dataset) // args.per_device_train_batch_size // args.gradient_accumulation_steps)
    num_train_epochs = max(math.ceil(num_train_epochs), 1)

    print(f"""
        Model: {args.model_name}
        Training epochs: {num_train_epochs}
        Number of training steps: {args.max_training_steps}
        Number of training batches: {len(train_dataset) // args.per_device_train_batch_size}
        Number of validation examples: {len(val_dataset)}
    """)

    # Check if total batch size 128
    # assert args.per_device_train_batch_size * args.gradient_accumulation_steps == 128

    training_args = Seq2SeqTrainingArguments(
        output_dir=os.path.join(args.output_dir, args.model_name),
        logging_steps=args.logging_steps,
        num_train_epochs=num_train_epochs,
        # max_steps=args.max_training_steps,
        optim="adafactor",
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_accumulation_steps=1,
        fp16=args.fp16,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.model_name,
        metric_for_best_model="bleu",
        save_total_limit=3,
        predict_with_generate=True,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        generation_config=model.base_model.generation_config,
        ddp_find_unused_parameters=False,
    )

    if args.report_to == 'wandb':
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

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.per_device_eval_batch_size,
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
                early_stopping=args.early_stopping,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                max_length=args.max_sequence_length,
                num_beams=args.num_beams,
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
    with open(os.path.join(args.output_dir, args.model_name, "val_predictions.txt"), "w") as f:
        all_predictions = [
            {
                "prediction": prediction,
                "reference": label[0]
            }
            for prediction, label in zip(val_predictions, val_labels) 
        ]

        json.dump(all_predictions, f)
        print(f'Predictions saved to {os.path.join(args.output_dir, args.model_name, "val_predictions.txt")}')

    val_bleu = sacrebleu.compute(predictions=val_predictions, references=val_labels)

    # Save scores json
    scores = {
        "val": val_bleu,
    }

    with open(os.path.join(args.output_dir, args.model_name, "va_scores.json"), "w") as f:
        json.dump(scores, f)
        print(f'Scores saved to {os.path.join(args.output_dir, args.model_name, "val_scores.json")}')




