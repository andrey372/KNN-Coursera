import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, jaccard_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
X_train = StandardScaler().fit_transform(X_train)
LR = LogisticRegression(C = 0.01, random_state=12, solver = 'lbfgs').fit(X_train, y_train)
yhat = LR.predict(X_test)
print(confusion_matrix(y_test, yhat), classification_report(y_test, yhat), LR.score(X_test, y_test))
