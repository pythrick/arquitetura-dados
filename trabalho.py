#%%
import warnings

warnings.filterwarnings("ignore")

#%%

# Modules import
from pprint import pprint

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    plot_confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from numpy.random import RandomState

#%%

RANDOM_NUM = 42
np.random.seed(42)
RANDOM_STATE = RandomState(42)

#%%

df = pd.read_csv("weight_lifting.csv", header=1)
df = df.drop(columns=["user_name", "new_window", "cvtd_timestamp"])
df.fillna(df.mean(), inplace=True)
# df.fillna(0, inplace=True)
df.dropna(inplace=True)

# df.head()
# df.groupby('classe').size()

#%%

columns = [
    "kurtosis_picth_belt",
    "kurtosis_yaw_belt",
    "skewness_roll_belt.1",
    "skewness_yaw_belt",
    "kurtosis_roll_arm",
    "kurtosis_picth_arm",
    "kurtosis_yaw_arm",
    "skewness_roll_arm",
    "skewness_pitch_arm",
    "skewness_yaw_arm",
    "kurtosis_yaw_dumbbell",
    "skewness_yaw_dumbbell",
    "kurtosis_roll_forearm",
    "kurtosis_picth_forearm",
    "kurtosis_yaw_forearm",
    "skewness_roll_forearm",
    "skewness_pitch_forearm",
    "skewness_yaw_forearm",
    "max_yaw_forearm",
    "min_yaw_forearm",
    "amplitude_yaw_forearm",
]
for col in columns:
    df[col] = df[col].astype(str)
    df[col] = df[col].str.replace("#DIV/0!", "0")

#%%

# Datasets
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,)


#%%

models_base = [
    {
        "name": "LR",
        "classifier": LogisticRegression(max_iter=4000),
        "parameters": [
            {
                "penalty": ["l2"],
                "solver": ["newton-cg", "sag", "lbfgs", "liblinear"],
                "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                "fit_intercept": [True, False],
                "multi_class": ["auto", "ovr"],
            },
            {
                "penalty": ["elasticnet"],
                "solver": ["saga"],
                "C": [0.001],  # 0.01, 0.1, 1, 10, 100, 1000
                "fit_intercept": [True, False],
                "multi_class": ["auto", "ovr", "multinomial"],
            },
        ],
    },
    {
        "name": "SVM",
        "classifier": SVC(gamma="scale", probability=True),
        "parameters": [
            {"C": [0.1, 0.5, 1, 10, 100, 500, 1000], "kernel": ["poly"]},
            {
                "C": [0.1, 0.5, 1, 10, 100, 500, 1000],
                "gamma": [0.1, 0.001, 0.0001, 0.00001],
                "kernel": ["rbf"],
            },
        ],
    },
    {
        "name": "MPL",
        "classifier": MLPClassifier(),
        "parameters": [
            {
                "solver": ["lbfgs", "sgd", "adam"],
                "alpha": [1e-4, 1e-5],
                "hidden_layer_sizes": [(5, 2), (100,)],
            }
        ],
    },
]


#%%

# TRAIN AND PREDICT
models_base_predict = []
for mb in models_base:
    model = GridSearchCV(mb["classifier"], mb["parameters"], n_jobs=-1, verbose=1)
    model = model.fit(X_train, np.ravel(y_train, order="C"))
    predict = model.predict(X_test)

    best_parameters = model.best_params_
    result = {
        "name": mb["name"],
        "best_score": model.best_score_,
        "best_parameters": best_parameters,
        "predict": predict,
        "model": model,
    }
    pprint(result)
    models_base_predict.append(result)


#%% md

### Evaluate predictions

#%%


def plot_results():
    for result in models_base_predict:
        print(f"Model: {result['name']}")
        print(f"Accuracy: {round(accuracy_score(y_test, result['predict']), 4)}")
        print()
        print(confusion_matrix(y_test, result["predict"]))
        print()
        print(classification_report(y_test, result["predict"]))
        print()
        plot_confusion_matrix(model, X_test, y_test)
        plt.show()
        print("--------------------------------------------")


#%%

plot_results()

#%%
