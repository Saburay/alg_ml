import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
import random
import warnings


class MyLogReg:

    def __init__(self, n_iter=100, learning_rate=0.1, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self):
        params = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{__class__.__name__} class: {params}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=False):
        X.insert(loc=0, column='col_0', value=1)
        self.weights = np.ones(X.shape[1])
        eps = 1e-15
        n = X.shape[0]
        for i in range(self.n_iter):
            y_pred = 1 / (1 + np.exp(-1 * (X.dot(self.weights))))
            LogLoss = (sum(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))) * (-1 / n)
            grad = ((1 / n) * ((y_pred - y).dot(X)))
            self.weights -= self.learning_rate * grad

            if verbose:
                s = 'start'
                if iter == 1 or iter % 10 == 0:
                    print(f'{s}|loss:{LogLoss}|')
                    s = iter

    def get_coef(self):
        return np.array(self.weights[1:])

    def predict(self, xx):
        xx.insert(loc=0, column='oness', value=1)
        pred_xx = 1 / (1 + np.exp(-1 * (xx.dot(self.weights.to_numpy()))))
        pred_xx = pred_xx.round().astype(int)
        return pred_xx

    def predict_proba(self, xy):
        pred_xy = 1 / (1 + np.exp(-1 * (xy.dot(self.weights.to_numpy()))))

        return pred_xy


X, y = make_regression(
n_samples=400,
n_features=14,
n_informative=5,
noise=15)
X = pd.DataFrame(X)
y = pd.Series(y)
#X, _, y, _ = train_test_split(X, y, test_size=0.2, random_state=42)

line = MyLogReg(n_iter=50, learning_rate=0.1)
line.fit(X, y, verbose=None)
#X, y = make_regression(n_samples=100, n_features=14, n_informative=10, noise=15, random_state=66)
# X = pd.DataFrame(X)
# y = pd.Series(y)
#X.columns = [f'col_{col}' for col in X.columns]
#print(line.__repr__())
print('SUM:--->  ',line.get_coef().sum())
