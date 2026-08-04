import csv
import sys
import yaml
import numpy as np
import pandas as pd
import evaluate
import argparse
from datasets import Dataset
from pathlib import Path
from transformers import XLNetTokenizer, XLNetForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score, classification_report,
                             confusion_matrix, roc_auc_score)

parser = argparse.ArgumentParser(
                    prog='BERT Inference',
                    description='Uses a finetuned model for inference',
                    epilog='Text at the bottom of help')
parser.add_argument(
    "--config",
    required=True,
    help="Path to the config file"
)
parser.add_argument("--model_path", "--model_path")
parser.add_argument("--seed", "--seed")
parser.add_argument("--mask", "--mask", type=str)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

MODEL_NAME = "bert-base-uncased"

def identify_classifications(y_true:list, y_pred:list) -> list:
    """
    Compare the predictions against the ground truth and retrieve the classifcation type
    :param y_true:
    :param y_pred:
    :return classifications:
    """
    classifications = []
    if len(y_true) != len(y_pred):
        raise ValueError("Lists do not have an equal length!")
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            classifications.append("TP")
        elif t == 0 and p == 1:
            classifications.append("FP")
        elif t == 1 and p == 0:
            classifications.append("FN")
        else:
            classifications.append("TN")
    return classifications


if __name__ == "__main__":
    print("#################################################### START #################################################\n")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    MODEL_PATH = args.model_path # provided by argparse, since this may change with seeds
    SEED = args.seed #
    MASKING = args.mask #config["masking_scheme"] # will be given by yaml
    DATASET = config["test_set_key"] # will be given by yaml
    testfile = config["test_set"]
    assert DATASET in testfile
    #assert config["model_name_or_path"] in testfile
    #assert config["model_name_or_path"] in config["tokenizer_name_or_path"]


    #filename = TEST_SET_MAP[DATASET]
    print("Model Path: ", MODEL_PATH)
    print("DATA: ", DATASET)
    print("TESTSET: ", testfile)
    print("MASKING: ", MASKING)

    outputfile = config["output_directory"] + SEED + "_" + DATASET + "_" + MASKING +".csv"
    Path(config["output_directory"]).mkdir(parents=True, exist_ok=True)
    tokenizer = XLNetTokenizer.from_pretrained(config["tokenizer_name_or_path"], max_len=512)
    model = XLNetForSequenceClassification.from_pretrained(MODEL_PATH)
    model.resize_token_embeddings(len(tokenizer)) # saved tokenizer has an additional token



    dataset_test = pd.read_csv(testfile,
                          delimiter='\t',quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                            lineterminator="\n")
    if DATASET == "pan_13":
        dataset_test = dataset_test.fillna('')
    dataset_test["document"] = dataset_test["document"].astype(str)

    test_dataset = Dataset.from_pandas(dataset_test)


    X_test = test_dataset.map(lambda e: tokenizer(e["document"], truncation=True, padding="max_length"), batched=True)
    X_test = X_test.shuffle(seed=42)


    training_args = TrainingArguments(output_dir="/", per_device_eval_batch_size=8, report_to=None)
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics
    )
    model.eval()
    res = trainer.predict(X_test)
    predictions = np.argmax(res.predictions, axis=-1)
    predictions = predictions.tolist()
    print("Accuracy: ", accuracy_score(X_test["label"], predictions), round(accuracy_score(X_test["label"], predictions), 4))
    print("Precision Binary: ", precision_score(X_test["label"], predictions, average=None))
    print("Recall Binary: ", recall_score(X_test["label"], predictions, average=None))
    print("F1 Score Binary: ", f1_score(X_test["label"], predictions, average=None))
    print("ROC AUC SCORE: ", roc_auc_score(X_test["label"], predictions))
    #print("Predictions: \n", predictions)
    #print("True: \n", X_test["label"])
    label_mapping = {
        0: "Male",
        1: "Female"
    }
    y_true_named = [label_mapping[x] for x in X_test["label"]]
    y_pred_named = [label_mapping[x] for x in predictions]
    print(classification_report(y_true_named, y_pred_named, labels=["Male", "Female"]))
    print(confusion_matrix(y_true_named, y_pred_named, labels=["Male", "Female"]))
    classifications = identify_classifications(X_test["label"], predictions)
    df = pd.DataFrame({
        "author_ids": X_test["author_id"],
        "y_true": X_test["label"],
        "y_pred": predictions,
        "classifications": classifications
    })
    df.to_csv(outputfile, index=False)
    print("#########################################################################################################\n")
    print("#########################################################################################################\n")
    # def write out predictions and error type

