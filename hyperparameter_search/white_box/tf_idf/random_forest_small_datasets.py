import datetime
import json
import sys
import csv
import numpy as np

import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib

data_with_60_20_20_split = [""]
data_with_80_20_split = [""]

def read_in_files(train, val):
    cols_t = train.columns.to_list()
    cols_v = val.columns.to_list()
    if cols_t != cols_v:
        raise ValueError("Columns not similar")
    features = cols_t[1]
    print("FEATURES")
    print(features)
    label = cols_t[-1]
    print("LABEL")
    print(label)
    X_train = train[features]
    y_train = train[label]
    X_val = val[features]
    y_val = val[label]
    return X_train, X_val, y_train, y_val


if __name__ == "__main__":
    n_jobs = int(sys.argv[1])
    filename = sys.argv[2]
    val_set = sys.argv[3]
    data_name = sys.argv[4]
    outputfile = f"rf_bl_{data_name}_metadata.json"
    model_and_vectorizer_file = f"rf_bl_{data_name}_best_model.json"

    dataset = pd.read_csv(filename,
                          delimiter='\t',quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                          lineterminator="\n")
    val_set = pd.read_csv(val_set,
                          delimiter='\t',quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                          lineterminator="\n")

    cols = list(dataset.columns)

    # Remove the columns you want to reposition
    cols.remove("document")
    cols.remove("label")

    # Build the new order
    new_cols = [cols[0], "document"] + cols[1:] + ["label"]

    dataset = dataset[new_cols]
    val_set = val_set[new_cols]

    X_train, X_val, y_train, y_val = read_in_files(dataset, val_set)

    X = np.concatenate([X_train, X_val])
    y = np.concatenate([y_train, y_val])

    print("DATASET: ", filename)
    print("VAL: ", val_set)

    test_fold = [-1] * len(X_train) + [0] * len(X_val)
    #X,y = create_80_20_split(dataset)
    ps = PredefinedSplit(test_fold)


    param_grid = [
    {
        "vect": [TfidfVectorizer()],
        "vect__ngram_range": [(1,1), (1,2)],
        "vect__max_df": [0.9],
        "vect__min_df": [1, 5],
        "vect__norm": ["l2"],
        "vect__sublinear_tf": [True],

        "rf__n_estimators": [300],
        "rf__max_depth": [None, 50],
        "rf__min_samples_leaf": [1, 4],
        "rf__max_features": ["sqrt"],
        "rf__bootstrap": [True],
        "rf__criterion": ["gini"],
    }
    ]

    pipe = Pipeline([
        ("vect", TfidfVectorizer()),
        ("rf", RandomForestClassifier(random_state=42))
    ])

    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring="accuracy",
        cv=ps,
        n_jobs=n_jobs,
        verbose=3 # the higher, the more messages, range from 1-3
    )
    with parallel_backend("loky"): # default is loky, single-host, process-based parallelism
        grid.fit(X,y)
        print(sorted(grid.cv_results_.keys()))
        print("Best parameters found: ", grid.best_params_)
        print("Best cross-validation score: ", grid.best_score_)

        best_params = {
            k: str(v)
            for k, v in grid.best_params_.items()
        }

        result = {
            "timestamp": datetime.datetime.now().isoformat(),
            "data": filename,
            "best_params": best_params,
            "best_score": grid.best_score_,
            "scoring": str(grid.scoring),
            "best_estimator": str(grid.best_estimator_),
        }
        with open(model_and_vectorizer_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)
        joblib.dump(grid.best_estimator_, model_and_vectorizer_file)




