import torch

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

# Add collate_fn to DataLoader
def collate_fn(batch):
    # Add padding to the inputs 
    # "inputs" must be 250 frames long
    # "attention_mask" must be 250 frames long
    # "labels" must be 128 tokens long
    return {
        "sign_inputs": torch.stack([
            torch.cat((sample["sign_inputs"], torch.zeros(250 - sample["sign_inputs"].shape[0], 208)), dim=0)
            for sample in batch
        ]),
        "attention_mask": torch.stack([
            torch.cat((sample["attention_mask"], torch.zeros(250 - sample["attention_mask"].shape[0])), dim=0)
            if sample["attention_mask"].shape[0] < 250
            else sample["attention_mask"]
            for sample in batch
        ]),
        "labels": torch.stack([
            torch.cat((sample["labels"].squeeze(0), torch.zeros(128 - sample["labels"].shape[0])), dim=0)
            if sample["labels"].shape[0] < 128
            else sample["labels"]
            for sample in batch
        ]).squeeze(0).to(torch.long),
    }
