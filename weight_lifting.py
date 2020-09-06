import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    plot_confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from numpy.random import RandomState
import warnings
from typing import List
import random

from utils import to_latex

warnings.filterwarnings("ignore")

random.seed(42)
RANDOM_NUM = 42
np.random.seed(42)
RANDOM_STATE = RandomState(42)


class WeightLifting:
    @staticmethod
    def load_df() -> pd.DataFrame:
        df = pd.read_csv("weight_lifting.csv", header=1)
        df.to_csv(r"outputs/original_database.csv", quoting=csv.QUOTE_NONNUMERIC)
        return df

    @staticmethod
    def transform(df: pd.DataFrame) -> pd.DataFrame:
        df.drop(columns=["user_name", "cvtd_timestamp",], inplace=True)

        # Convertendo coluna "new_window" para booleano
        df["new_window"] = np.where(df["new_window"].str.lower() == "yes", 1, 0)
        df["new_window"] = df["new_window"].astype(int)

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

        df.to_csv(r"outputs/cleaned_database.csv", quoting=csv.QUOTE_NONNUMERIC)
        return df

    @staticmethod
    def create_train_test(
        df: pd.DataFrame,
        features: List[str] = None,
        target: str = None,
        test_size: float = 0.25,
    ) -> List:
        if features is None:
            X = df.iloc[:, 0:-1]
        else:
            X = df[features]

        if target is None:
            y = df.iloc[:, -1:]
        else:
            y = df[target]

        return train_test_split(X, y, test_size=test_size, random_state=RANDOM_NUM)

    @staticmethod
    def fit_and_predict(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        list_model: List = None,
        state: str = "INICIAL",
    ) -> List:
        if list_model is None:
            list_model = [
                (
                    "LR",
                    LogisticRegression(
                        **{
                            "C": 0.1,
                            "fit_intercept": True,
                            "multi_class": "ovr",
                            "penalty": "l2",
                            "solver": "newton-cg",
                        }
                    ),
                ),
                (
                    "SVM",
                    SVC(
                        **{
                            "C": 10,
                            "gamma": 1e-05,
                            "kernel": "rbf",
                            "probability": True,
                        }
                    ),
                ),
                (
                    "MPL",
                    MLPClassifier(
                        **{
                            "alpha": 0.0001,
                            "hidden_layer_sizes": (5, 2),
                            "solver": "sgd",
                        }
                    ),
                ),
            ]
        models_base_predict = []
        for result in list_model:
            name, model = result
            model.fit(X_train, y_train)
            predict = model.predict(X_test)
            accuracy = round(accuracy_score(y_test, predict), 4)
            f1 = round(f1_score(y_test, predict, average="macro"), 4)
            precision = round(precision_score(y_test, predict, average="macro"), 4)
            recall = round(recall_score(y_test, predict, average="macro"), 4)
            models_base_predict.append(
                {
                    "name": name,
                    "state": state,
                    "model": model,
                    "predict": predict,
                    "accuracy": accuracy,
                    "f1": f1,
                    "precision": precision,
                    "recall": recall,
                    "state_name": f"{state}_{name}",
                }
            )

        return models_base_predict

    @staticmethod
    def plot_results(list_predict, X_test, y_test, export_files=True):
        for result in list_predict:
            print(f"Model: {result['name']}")
            metrics = {
                "Accuracy": [result["accuracy"]],
                "F1": [result["f1"]],
                "Precision": [result["precision"]],
                "Recall": [result["recall"]],
            }

            metrics_df = pd.DataFrame.from_dict(
                metrics, orient="index", columns=["Valor"],
            )
            print(metrics_df)
            print()
            print(confusion_matrix(y_test, result["predict"]))
            print()
            report = classification_report(y_test, result["predict"], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            print(report_df)
            plot_confusion_matrix(result["model"], X_test, y_test)
            print()
            if export_files:
                # to_latex(
                #     metrics_df,
                #     f"outputs/tex/table_metrics_{result['state_name'].lower()}.tex",
                #     float_format="%.2f",
                # )
                to_latex(
                    report_df,  # report_df.iloc[:-3, :-1],
                    f"outputs/tex/table_{result['state_name'].lower()}.tex",
                    float_format="%.2f",
                )
                plt.savefig(f"outputs/img/matrix_{result['state_name'].lower()}.png")
            plt.show()
            print("--------------------------------------------")

    @staticmethod
    def plot_correlation_matrix(df: pd.DataFrame):
        corr = df.corr()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax)
        ticks = np.arange(0, len(df.columns), 1)
        ax.set_xticks(ticks)
        plt.xticks(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(df.columns)
        ax.set_yticklabels(df.columns)
        plt.show()

    @staticmethod
    def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
        # identify and remove outliers from dataframe
        iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE)
        predict = iso.fit_predict(df.iloc[:, 0:-1])

        mask = predict != -1
        return df.iloc[mask]

    def plot_final_results(self, df: pd.DataFrame):
        # print(df)

        def get_approach(row: pd.Series):
            return {
                "INICIAL": "Inicial",
                "ISO": "Floresta de Isolamento",
                "SFS": "Sequential Feature Selector",
                "ISO_SFS": "SFS + Floresta de Isolamento",
            }.get(row["state"])

        def get_approach_order(row: pd.Series):
            return {"INICIAL": 0, "ISO": 1, "SFS": 2, "ISO_SFS": 3,}.get(row["state"])

        def get_classifier_order(row: pd.Series):
            return {"LR": 0, "SVM": 1, "MPL": 2,}.get(row["name"])

        def get_classifier(row: pd.Series):
            return {
                "LR": "Regressão Logística",
                "SVM": "Máquina de Vetores de Suporte",
                "MPL": "Perceptron Multicamadas",
            }.get(row["name"])

        df["tecnica"] = df.apply(lambda x: get_approach(x), axis=1)
        df["classificador"] = df.apply(lambda x: get_classifier(x), axis=1)
        df["ordem_tecnica"] = df.apply(lambda x: get_approach_order(x), axis=1)
        df["ordem_classificador"] = df.apply(lambda x: get_classifier_order(x), axis=1)
        df = df.sort_values(by=["ordem_classificador", "ordem_tecnica"])
        resultados_df = df[
            ["classificador", "tecnica", "accuracy", "f1", "precision", "recall"]
        ]
        to_latex(resultados_df, "outputs/tex/table_resultado_final.tex", index=False)
        print(resultados_df)

        sns.set()
        columns = [
            "name",
            "state",
            "state_name",
            "accuracy",
            "f1",
            "precision",
            "recall",
        ]
        resultados = df[columns]
        resultados.columns = [col.lower() for col in columns]
        resultados.to_csv("outputs/resultados.csv")
        pivot = resultados.pivot("name", "state", "accuracy")[
            ["INICIAL", "ISO", "SFS", "ISO_SFS"]
        ]
        sns_plot = sns.heatmap(pivot, annot=True, linewidths=0.5)
        sns_plot.figure.savefig("outputs/img/results_heatmap.png")
