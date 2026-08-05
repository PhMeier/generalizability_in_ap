import csv
import json
import numpy as np
import pandas as pd
import evaluate
from ray import tune
import argparse
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import XLNetTokenizer, XLNetModel, XLNetForSequenceClassification, TrainingArguments, Trainer

MODEL_NAME = "xlnet-base-cased"


parser = argparse.ArgumentParser(
                    prog='XL Hyp',
                    description='Hyperparameter search for a dataset',
                    epilog='Text at the bottom of help')
parser.add_argument(
    "--n_trials",
    required=True,
    help="Path to the config file",
    type=int,
    default=60
)
parser.add_argument("--output_directory", "--output_directory")
parser.add_argument("--training_file", "--training_file")
parser.add_argument("--val_file", "--val_file")






def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def compute_objective(metrics):
    return metrics["eval_accuracy"]



def tokenize_batch(batch):
    return tokenizer(
        batch["document"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )


# Init for hyperparameter search
def model_init():
    model = XLNetForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    )
    model.resize_token_embeddings(len(tokenizer))
    return model


def hp_space_optuna(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 5e-7, 2e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical(
            "per_device_train_batch_size", [8,16]
        ),
        "gradient_accumulation_steps": trial.suggest_categorical(
            "gradient_accumulation_steps", [1, 2, 4, 8]
        ),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0,0.01,0.05]),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.05, 0.2),
        "lr_scheduler_type": trial.suggest_categorical(
            "lr_scheduler_type", ["linear", "cosine"]
        ),
        "num_train_epochs": trial.suggest_categorical(
            "num_train_epochs", [3, 4, 5, 6, 8]
        ),
    }

if __name__ == "__main__":
    args = parser.parse_args()
    output_directory = args.output_directory
    training_file = args.training_file
    val_file = args.val_file
    assert "xlnet" in training_file
    assert "xlnet" in val_file
    N_TRIALS = args.n_trials #60
    print(f"Number of Trals: {N_TRIALS}")
    print(f"Training File: {training_file}")
    print(f"Validation File: {val_file}")
    OUTPUT_DIR = f"./xlnet/{output_directory}/"
    LOG_DIR = f"./xlnet/{output_directory}/logs"
    print(OUTPUT_DIR)
    tokenizer = XLNetTokenizer.from_pretrained(MODEL_NAME, max_length=512, max_len=512)
    # max len updatesd the maximum length of XLNET
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["[URL]"]}
    )

    dataset_train = pd.read_csv(training_file,
                          delimiter='\t',quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                            lineterminator="\n")

    dataset_val = pd.read_csv(val_file,
                          delimiter='\t',quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                            lineterminator="\n")


    train_dataset = Dataset.from_pandas(dataset_train)
    val_dataset = Dataset.from_pandas(dataset_val)

    # changed from lambda function due to debugging issue
    #dataset_train = train_dataset.map(lambda e: tokenizer(e["text"], truncation=True, padding="max_length"), batched=True)
    dataset_train = train_dataset.map(tokenize_batch, batched=True)
    dataset_train = dataset_train.shuffle(seed=42)
    #dataset_val = val_dataset.map(lambda e: tokenizer(e["text"], truncation=True, padding="max_length"), batched=True)
    dataset_val = val_dataset.map(tokenize_batch, batched=True)
    dataset_val = dataset_val.shuffle(seed=42)




    # Training arguments, these are the default configurations
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOG_DIR,
        report_to=["tensorboard"],
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        do_eval=True,
        load_best_model_at_end=False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        gradient_accumulation_steps=1,
        save_total_limit=2,
        warmup_ratio=0.1,
        weight_decay=0.0,
        lr_scheduler_type="linear",
        learning_rate=1e-5,
        eval_strategy="epoch",
        save_strategy="no",
        seed=42,
        fp16=True,
    )

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        compute_metrics=compute_metrics
    )

    best_run = trainer.hyperparameter_search(
        backend="optuna", #"ray",
        hp_space=hp_space_optuna,
        direction="maximize",
        n_trials=N_TRIALS,
        compute_objective=compute_objective,
    )
    print("Best Run")
    print(best_run)
    print("Best hyperparameters:")
    print(best_run.hyperparameters)
    #trainer.train()
    #trainer.save_model(OUTPUT_DIR)


    best_params = best_run.hyperparameters
    tokenizer.save_pretrained(OUTPUT_DIR + "tokenizer/")
    with open(OUTPUT_DIR+"best_hyperparameters.json", "w") as f:
        json.dump(best_run.hyperparameters, f, indent=4)

    final_args = TrainingArguments(
        output_dir=OUTPUT_DIR + "/best_model",
        logging_dir=LOG_DIR,
        report_to=["tensorboard"],
        do_eval=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        seed=42,
        fp16=True,
        **best_params,
    )

    final_trainer = Trainer(
        model_init=model_init,
        args=final_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
        compute_metrics=compute_metrics,
    )