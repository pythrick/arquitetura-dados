#%%

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    plot_confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

#%%

random = 42
np.random.seed(42)

#%%

df = pd.read_csv("weight_lifting.csv", header=1)
df = df.drop(columns=['user_name', 'new_window','cvtd_timestamp'])
df = df.fillna(0)
#df.head()
#df.groupby('classe').size()

#%%

columns = ['kurtosis_picth_belt', 'kurtosis_yaw_belt', 'skewness_roll_belt.1',
           'skewness_yaw_belt', 'kurtosis_roll_arm', 'kurtosis_picth_arm',
           'kurtosis_yaw_arm', 'skewness_roll_arm', 'skewness_pitch_arm',
           'skewness_yaw_arm', 'kurtosis_yaw_dumbbell', 'skewness_yaw_dumbbell',
           'kurtosis_roll_forearm', 'kurtosis_picth_forearm', 'kurtosis_yaw_forearm',
           'skewness_roll_forearm', 'skewness_pitch_forearm', 'skewness_yaw_forearm',
           'max_yaw_forearm', 'min_yaw_forearm', 'amplitude_yaw_forearm']
for col in columns:
    df[col] = df[col].astype(str)
    df[col] = df[col].str.replace('#DIV/0!', '0')

#%%

# Datasets
X = df.iloc[:, 0:-1]
y = df.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#%%

# MODELS
models_base = []
models_base.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models_base.append(('SVM', SVC(gamma='auto')))
models_base.append(('MPL', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42)))

#%%

models_base = [
    {
        "name": "LR",
        "pipeline": Pipeline([('clf', LogisticRegression)]),
        "parameters": {
            "clf__solver": []
        }
    }
]

#%%

# CROSS VAL SCORE
for name, model in models_base:
    kfold = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy', n_jobs=3)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#%%

# TRAIN AND PREDICT
models_base_predict = []
for result in models_base:
    name, model = result
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    models_base_predict.append((name, model, predict))

#%% md

### Evaluate predictions

#%%

def plot_results():
    for result in models_base_predict:
        name, model, predict = result
        print(f'Model: {name}')
        print(f' Accuracy: {round(accuracy_score(y_test, predict), 4)}')
        print()
        print(confusion_matrix(y_test, predict))
        print()
        print(classification_report(y_test, predict))
        print()
        plot_confusion_matrix(model, X_test, y_test)
        plt.show()
        print('--------------------------------------------')

#%%

plot_results()

#%%


