import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, AutoTokenizer
import numpy as np
import yaml
import wandb

config_file = "config.yaml"

with open(config_file, "r") as file:
    config = yaml.safe_load(file)

# Load config values
dsn = config["TTS_dataset"]
model_name = config["model_name"]
run_name = config["run_name"]
project_name = config["project_name"]
base_repo_id = config["save_folder"]
epochs = config["epochs"]
batch_size = config["batch_size"]
save_steps = config["save_steps"]
pad_token = config["pad_token"]
number_processes = config["number_processes"]
learning_rate = config["learning_rate"]

# LoRA configuration
lora_rank = 32
lora_alpha = 64
lora_dropout = 0.0

def train_fn(rank):
    device = xm.xla_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
        bias="none",
        modules_to_save=["lm_head", "embed_tokens"],
        task_type="CAUSAL_LM",
        use_rslora=True,
    )

    model = get_peft_model(model, lora_config)

    ds = load_dataset(dsn, split="train")

    wandb.init(project=project_name, name=f"{run_name}-{rank}")

    training_args = TrainingArguments(
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        logging_steps=1,
        bf16=True,
        output_dir=f"./{base_repo_id}",
        report_to="wandb",
        save_steps=save_steps,
        remove_unused_columns=True,
        learning_rate=learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer
    )

    trainer.train()

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(f"./{base_repo_id}/merged")
    tokenizer.save_pretrained(f"./{base_repo_id}/merged")

if __name__ == '__main__':
    xmp.spawn(train_fn, args=(), nprocs=number_processes, start_method='fork')
