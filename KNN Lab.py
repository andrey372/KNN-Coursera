import itertools
import wget
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('teleCust1000t.csv')
X = df[[
    'region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ',
    'retire', 'gender', 'reside'
]].values
y = df['custcat'].values

#Normilize data
X = preprocessing.StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)

#Classification
k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
yhat_train = neigh.predict(X_train)
yhat_test = neigh.predict(X_test)

print(
    metrics.accuracy_score(y_train, yhat_train),
    metrics.accuracy_score(y_test, yhat_test),
    metrics.classification_report(y_test, yhat_test),
    metrics.confusion_matrix(y_test, yhat_test), neigh.score(X_test, y_test))
