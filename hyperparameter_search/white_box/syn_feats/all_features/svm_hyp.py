import sys
import csv
import json
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix

data_with_60_20_20_split = [""]
data_with_80_20_split = [""]

def read_in_files(train, val):
    cols_t = train.columns.to_list()
    cols_v = val.columns.to_list()
    start = cols_t.index("flesch_kincaid_easiness")
    end = -1

    features = cols_t[start:end]
    label = cols_t[end]
    print("Used Features")
    print(features)
    assert features[0] == "flesch_kincaid_easiness"
    X_train = train[features]
    y_train = train[label]
    X_val = val[features]
    y_val = val[label]
    return  X_train, X_val, y_train, y_val


if __name__ == "__main__":
    n_jobs = int(sys.argv[1])
    filename = sys.argv[2]
    val_set = sys.argv[3]
    data_name = sys.argv[4]


    dataset = pd.read_csv(filename,
                          delimiter='\t',quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                          lineterminator="\n", index_col=False)
    val_set = pd.read_csv(val_set,
                          delimiter='\t',quotechar='"', escapechar='\\', quoting=csv.QUOTE_ALL,
                          lineterminator="\n", index_col=False)


    dataset = dataset.drop("Unnamed: 0", errors="ignore")
    val_set = val_set.drop("Unnamed: 0", errors="ignore")
    #outputfile = f"svm_{data_name}.json"
    outputfile = f"svm_str_{data_name}.json"



    X_train, X_val, y_train, y_val = read_in_files(dataset, val_set)
    print("DATASET: ", filename)
    print("VAL: ", val_set)

    X = np.concatenate([X_train, X_val])
    y = np.concatenate([y_train, y_val])

    test_fold = [-1] * len(X_train) + [0] * len(X_val)
    #X,y = create_80_20_split(dataset)
    ps = PredefinedSplit(test_fold)

    param_grid = [{
        "svc__kernel":["rbf"],
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", 0.01, 0.001, 0.0001],
    }, {
        "svc__kernel":["linear"],
        "svc__C": [0.1, 1, 10, 100],
    }, {
        "svc__kernel":["poly"],
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": ["scale", 0.01, 0.001],
        "svc__degree": [2,3],
        "svc__coef0": [0,1],
    }]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(random_state=42))
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
            "timestamp": datetime.now().isoformat(),
            "data": filename,
            "best_params": grid.best_params_,
            "best_score": grid.best_score_,
            "scoring": str(grid.scoring),
            "best_estimator": str(grid.best_estimator_),
        }
        with open(outputfile, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4)








