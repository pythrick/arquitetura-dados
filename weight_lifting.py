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
import warnings
from typing import List
import random

warnings.filterwarnings("ignore")

random.seed(42)
RANDOM_NUM = 42
np.random.seed(42)
RANDOM_STATE = RandomState(42)

class WeightLifting():
    def load_df(self) -> pd.DataFrame:
        df = pd.read_csv("weight_lifting.csv", header=1)
        df.to_csv(r'outputs/original_database.csv', quoting=csv.QUOTE_NONNUMERIC)
        return df


    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns=["user_name", "cvtd_timestamp",], inplace=True)

        # Convertendo coluna "new_window" para booleano
        df['new_window'] = np.where(
            df['new_window'].str.lower() == 'yes', 1, 0)
        df['new_window'] = df['new_window'].astype(int)

        for col in df.columns[:-1]:
            # Corrigindo campos com "#DIV/0!"
            if df[col].dtype == object:
                df[col] = df[col].str.replace("#DIV/0!", "0")
                df[col] = df[col].astype(float)

            # Corrigindo valores N/A com a média ou "0"
            if df[col].dtype in (int, float):
                df[col] = df[col].replace(np.nan, df[col].mean())
            else:
                df[col] = df[col].replace(np.nan, "0")

        df.to_csv(r'outputs/cleaned_database.csv', quoting=csv.QUOTE_NONNUMERIC)
        return df


    def createTrainTest(
        self,
        df: pd.DataFrame,
        features: List[str] = None,
        target: str = None,
        test_size: float = 0.25):
        if features is None:
            X = df.iloc[:, 0:-1]
        else:
            X = df[features]

        if target is None:
            y = df.iloc[:, -1:]
        else:
            y = df[target]

        return train_test_split(X, y, test_size=test_size, random_state=RANDOM_NUM)


    def fitAndPredict(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        list_model: List= None) -> List:
        if list_model is None:
            list_model = [
                ('LR', LogisticRegression(**{'C': 0.1,
                                'fit_intercept': True,
                                'multi_class': 'ovr',
                                'penalty': 'l2',
                                'solver': 'newton-cg'})),
                ('SVM', SVC(**{'C': 10,
                                'gamma': 1e-05,
                                'kernel': 'rbf',
                                'probability': True})),
                ('MPL', MLPClassifier(**{'alpha': 0.0001,
                                'hidden_layer_sizes': (5, 2),
                                'solver': 'sgd'}))]
        models_base_predict = []
        for result in list_model:
            name, model = result
            model.fit(X_train, y_train)
            predict = model.predict(X_test)
            models_base_predict.append({
                "name": name,
                "model": model,
                "predict": predict
            })

        return models_base_predict


    def plot_results(
        self,
        list_predict,
        X_test,
        y_test):
        for result in list_predict:
            name, model, predict = result.values()
            print(f"Model: {name}")
            print(f"Accuracy: {round(accuracy_score(y_test, predict), 4)}")
            print(f"F1: {round(f1_score(y_test, predict, average='macro'), 4)}")
            print(f"Precision: {round(precision_score(y_test, predict, average='macro'), 4)}")
            print(f"Recall: {round(recall_score(y_test, predict, average='macro'), 4)}")
            print()
            print(confusion_matrix(y_test, predict))
            print()
            print(classification_report(y_test, predict))
            print()
            plot_confusion_matrix(model, X_test, y_test)
            plt.show()
            print("--------------------------------------------")