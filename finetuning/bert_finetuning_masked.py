import csv
import sys
import json
import yaml
import numpy as np
import pandas as pd
import evaluate
import argparse
from datasets import Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

parser = argparse.ArgumentParser(
    description="Finetune bert"
)
parser.add_argument(
    "--config",
    required=True,
    help="Path to the config file"
)
parser.add_argument(
    "--seed",
    required=True,
    help="Seed for model"
)
parser.add_argument(
    "--mask_type",
    required=True,
    help="Seed for model"
)


metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

MODEL_NAME = "bert-base-uncased"

if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    mask_type = args.mask_type
    # check if the masking is correct
    assert mask_type == config["mask_type"]
    assert mask_type in args.config
    assert mask_type in config["train_file"]
    assert mask_type in config["val_file"]

    SEED = int(args.seed)
    print("SEED: ", SEED)
    with open(config["hyperparameter_file"], "r") as f:
        best_params = json.load(f)
    OUTPUT_DIR = config["output_path"] + f"/bert_{SEED}_{mask_type}/"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    LOG_DIR = f"../pan_14/bert_{SEED}_{mask_type}/logs"
    assert config["model_name_or_path"] == MODEL_NAME # check if everything is correct

    tokenizer = BertTokenizer.from_pretrained(config["tokenizer_name_or_path"], max_len=512)
    model = BertForSequenceClassification.from_pretrained(config["model_type"])
    model.resize_token_embeddings(len(tokenizer))

    # check if we picked the correct training and val file
    assert "BERT" in config["train_file"]
    assert "BERT" in config["val_file"]
    print("Train")
    print(config["train_file"])
    print("Val")
    print(config["val_file"])

    dataset_train = pd.read_csv(config["train_file"], #train.tsv",
                          delimiter='\t',quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                            lineterminator="\n")

    dataset_val = pd.read_csv(config["val_file"], #validation.tsv",
                          delimiter='\t',quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                            lineterminator="\n")


    train_dataset = Dataset.from_pandas(dataset_train)
    val_dataset = Dataset.from_pandas(dataset_val)


    dataset_train = train_dataset.map(lambda e: tokenizer(e["masked_text"], truncation=True, padding="max_length"), batched=True)
    dataset_train = dataset_train.shuffle(seed=42)
    dataset_val = val_dataset.map(lambda e: tokenizer(e["masked_text"], truncation=True, padding="max_length"), batched=True)
    dataset_val = dataset_val.shuffle(seed=42)

# {'learning_rate': 1.1221839351201136e-05, 'per_device_train_batch_size': 8, 'gradient_accumulation_steps': 1, 'weight_decay': 0.0002040621024410714, 'warmup_ratio': 0.06944505711252645, 'lr_scheduler_type': 'cosine', 'num_train_epochs': 8}
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOG_DIR,
        report_to=["tensorboard"],
        per_device_train_batch_size=best_params["per_device_train_batch_size"],
        per_device_eval_batch_size=8,
        num_train_epochs=best_params["num_train_epochs"],
        do_eval=True,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
        save_total_limit=1,
        warmup_ratio=best_params["warmup_ratio"],
        weight_decay=best_params["weight_decay"],
        lr_scheduler_type=best_params["lr_scheduler_type"],
        learning_rate=best_params["learning_rate"],
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=SEED,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
