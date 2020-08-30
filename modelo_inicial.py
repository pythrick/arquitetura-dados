#!/usr/bin/env python
# coding: utf-8

# In[75]:


# Modules import
import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pprint import pprint

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    plot_confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from numpy.random import RandomState


# In[76]:


# Supressão de warnings
import warnings

warnings.filterwarnings("ignore")


# In[77]:


# Aumenta o número de linhas para visualização
pd.set_option('display.max_rows', 200)


# In[78]:


RANDOM_NUM = 42
np.random.seed(42)
RANDOM_STATE = RandomState(42)


# In[79]:


df = pd.read_csv("weight_lifting.csv", header=1)
df.to_csv(r'outputs/original_database.csv', quoting=csv.QUOTE_NONNUMERIC)
df.head()


# ### Limpeza mínima da base de dados

# In[80]:


df.drop(columns=["user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window"], inplace=True)
# df.dtypes


# In[81]:


# Corrigindo campos com "#DIV/0!"
for col in df.columns:
    if df[col].dtype == object and col != "classe":
        df[col] = df[col].str.replace("#DIV/0!", "0")
        df[col] = df[col].astype(float)
# df.dtypes


# In[82]:


# Corrigindo valores N/A com a média
df.fillna(df.mean(), inplace=True)


# In[83]:


df.to_csv(r'outputs/cleaned_database.csv', quoting=csv.QUOTE_NONNUMERIC)


# ### Divisão entre base de treino e teste

# In[84]:


# Datasets
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,)


# ### Descobrir os melhores parâmetros para os classificadores

# In[71]:


models_base = [
    {
        "name": "LR",
        "classifier": LogisticRegression(),
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
                "C": [0.001],
                "fit_intercept": [True, False],
                "multi_class": ["auto", "ovr", "multinomial"],
                "l1_ratio": [0, 0.5, 1],
            },
        ],
    },
    {
        "name": "SVM",
        "classifier": SVC(),
        "parameters": [
            {
                "C": [0.1, 0.5, 1, 10, 100, 500, 1000],
                "gamma": ["scale"],
                "kernel": ["poly"],
                "probability": [True],
            },
            {
                "C": [0.1, 0.5, 1, 10, 100, 500, 1000],
                "gamma": [0.1, 0.001, 0.0001, 0.00001],
                "kernel": ["rbf"],
                "probability": [True],
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


# In[69]:


# Treinar e predizer com GridSearchCV
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


# ### Treinar e predizer com os parâmetros ajustados

# In[85]:


models_base = [
    ('LR', LogisticRegression(**{
        'C': 0.01,
        'fit_intercept': True,
        'multi_class': 'ovr',
        'penalty': 'l2',
        'solver': 'newton-cg'})),
    ('SVM', SVC(**{
        'C': 10,
        'gamma': 1e-05,
        'kernel': 'rbf',
        'probability': True})),
    ('MPL', MLPClassifier(**{
        'alpha': 0.0001,
        'hidden_layer_sizes': (100,),
        'solver': 'adam'}))
]
models_base_predict = []
for result in models_base:
    name, model = result
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    models_base_predict.append({
        "name": name,
        "model": model,
        "predict": predict
    })


# ### Avaliar predições

# In[86]:



def plot_results():
    for result in models_base_predict:
        print(f"Model: {result['name']}")
        print(f"Accuracy: {round(accuracy_score(y_test, result['predict']), 4)}")
        print(f"F1: {round(f1_score(y_test, result['predict'], average='macro'), 4)}")
        print(f"Precision: {round(precision_score(y_test, result['predict'], average='macro'), 4)}")
        print(f"Recall: {round(recall_score(y_test, result['predict'], average='macro'), 4)}")
        print()
        print(confusion_matrix(y_test, result["predict"]))
        print()
        print(classification_report(y_test, result["predict"]))
        print()
        plot_confusion_matrix(model, X_test, y_test)
        plt.show()
        print("--------------------------------------------")


# In[87]:


plot_results()


# In[ ]:




