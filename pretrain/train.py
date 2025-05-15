import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from datasets import load_dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import yaml
import wandb
from huggingface_hub import HfApi

# Load config
config_file = "config.yaml"
with open(config_file, "r") as file:
    config = yaml.safe_load(file)

# Config params
dsn1 = config["text_QA_dataset"]
dsn2 = config["TTS_dataset"]
model_name = config["model_name"]
tokenizer_name = config["tokenizer_name"]
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]
config_ratio = config["ratio"]

# Dataset Class
class BatchedRatioDataset(Dataset):
    def __init__(self, dataset1, dataset2, batch_total, ratio=config_ratio):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_total = batch_total
        self.ratio = ratio

        num_cycles_ds1 = len(dataset1) // (batch_total * ratio)
        num_cycles_ds2 = len(dataset2) // batch_total
        self.num_cycles = min(num_cycles_ds1, num_cycles_ds2)

        self.length = self.num_cycles * (ratio + 1) * batch_total

    def __len__(self):
        return int(self.length)

    def __getitem__(self, index):
        cycle_length = (self.ratio + 1) * self.batch_total
        cycle = index // cycle_length
        pos_in_cycle = index % cycle_length

        if pos_in_cycle < self.ratio * self.batch_total:
            batch_in_cycle = pos_in_cycle // self.batch_total
            sample_in_batch = pos_in_cycle % self.batch_total
            ds1_index = cycle * self.ratio * self.batch_total + batch_in_cycle * self.batch_total + sample_in_batch
            return self.dataset1[ds1_index]
        else:
            sample_in_batch = pos_in_cycle - self.ratio * self.batch_total
            ds2_index = cycle * self.batch_total + sample_in_batch
            return self.dataset2[ds2_index]

# Data collator
def data_collator(features):
    input_ids = [f["input_ids"] for f in features]
    attention_mask = [f.get("attention_mask", [1]*len(f["input_ids"])) for f in features]
    labels = [f.get("labels", f["input_ids"]) for f in features]

    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(i) for i in input_ids], batch_first=True, padding_value=pad_token)
    attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(m) for m in attention_mask], batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(l) for l in labels], batch_first=True, padding_value=-100)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# Main process
def train_fn(rank):
    device = xm.xla_device()

    wandb.init(project=project_name, name=f"{run_name}-{rank}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    new_tokens = [f"<custom_token_{i}>" for i in range(7 * 4096 + 11)]
    tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))

    ds1 = load_dataset(dsn1, split="train")
    ds2 = load_dataset(dsn2, split="train")
    batch_total = batch_size * number_processes
    train_dataset = BatchedRatioDataset(ds1, ds2, batch_total, ratio=config_ratio)

    training_args = TrainingArguments(
        output_dir=f"./{base_repo_id}",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=1,
        bf16=True,
        save_steps=save_steps,
        report_to="wandb",
        remove_unused_columns=True,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()

# Entry point
if __name__ == '__main__':
    xmp.spawn(train_fn, args=(), nprocs=number_processes, start_method='fork')
