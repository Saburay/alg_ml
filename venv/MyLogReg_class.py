import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import random


class MyLogReg:
    def __init__(self,
                 n_iter: int = 10,
                 learning_rate=0.1,
                 metric: str = None,
                 reg: str = 'None',
                 l1_coef: float = 0.,
                 l2_coef: float = 0.,
                 sgd_sample: float = None,
                 random_state: int = 42) -> None:
        self.score = None
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self._weights = None
        self.metric = metric  # accuracy precision recall f1 roc_auc
        self.reg = reg  # 'l1', 'l2', 'elasticnet'
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.sgd_sample = sgd_sample
        random.seed(random_state)

    def __str__(self):
        return f'MyLogReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def _loss(self, y_true: pd.Series, y_pred: pd.DataFrame) -> float:
        if self.reg == 'None':
            return log_loss(y_true, y_pred)
        elif self.reg == 'l1':
            return log_loss(y_true, y_pred) + \
                   self.l1_coef * self._weights.abs().sum()
        elif self.reg == 'l2':
            return log_loss(y_true, y_pred) + \
                   self.l2_coef * self._weights.pow(2).sum()
        else:
            return log_loss(y_true, y_pred) + \
                   self.l1_coef * self._weights.abs().sum() + \
                   self.l2_coef * self._weights.pow(2).sum()

    def _grad(self,
              y_true: pd.Series,
              y_pred: pd.DataFrame,
              X: pd.DataFrame,
              mini_batch_idx: pd.DataFrame.index) -> pd.Series:
        y_true = y_true.iloc[mini_batch_idx]
        y_pred = y_pred.iloc[mini_batch_idx]
        X = X.iloc[mini_batch_idx, :]

        if self.reg == 'None':
            return (y_pred - y_true) @ X / y_true.shape[0]
        elif self.reg == 'l1':
            return (y_pred - y_true) @ X / y_true.shape[0] + \
                   self.l1_coef * np.sign(self._weights)
        elif self.reg == 'l2':
            return (y_pred - y_true) @ X / y_true.shape[0] + \
                   self.l2_coef * 2 * self._weights
        else:
            return (y_pred - y_true) @ X / y_true.shape[0] + \
                   self.l1_coef * np.sign(self._weights) + \
                   self.l2_coef * 2 * self._weights

    @staticmethod
    def _get_minibatch_idx(x_length: int, sample) -> list:
        sample = int(sample * x_length) if isinstance(sample, float) else sample
        return random.sample(range(x_length), sample)

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = False) -> None:
        X = pd.concat([pd.DataFrame([1] * X.shape[0], index=X.index), X], axis=1)
        self._weights = pd.Series([1.] * X.shape[1], index=X.columns)
        for i in range(self.n_iter):
            minibatch_idx = self._get_minibatch_idx(X.shape[0],
                                                    self.sgd_sample) if self.sgd_sample else range(X.shape[0])
            y_hat = self.predict_proba(X.iloc[:, 1:])
            loss = self._loss(y, y_hat)
            grad = self._grad(y, y_hat, X, minibatch_idx)
            if isinstance(self.learning_rate, (int, float)):
                self._weights -= self.learning_rate * grad
            elif callable(self.learning_rate):
                self._weights -= self.learning_rate(i + 1) * grad
            if self.metric and self.metric != 'roc_auc':
                self.score = eval(self.metric)(y, self.predict(X.iloc[:, 1:]))
            elif self.metric == 'roc_auc':
                self.score = roc_auc(y, self.predict_proba(X.iloc[:, 1:]))
            if verbose and i % verbose == 0:
                if self.metric:
                    print(f'{i} | loss: {loss} | {self.metric}: {self.score}')
                else:
                    print(f'{i} | loss: {loss}')

    def get_coef(self):
        return self._weights[1:]

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.concat([pd.DataFrame([1] * X.shape[0], index=X.index), X], axis=1)
        return 1 / (1 + np.exp(- self._weights @ X.T))

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return (self.predict_proba(X) > 0.5).astype(int)

    def get_best_score(self):
        return self.score


def log_loss(y_true: pd.Series, y_pred: pd.DataFrame) -> float:
    return -(y_true * np.log(y_pred) + (1 + y_true) * np.log(1 - y_pred)).mean()


def f1(y_true: pd.Series, y_pred: pd.DataFrame) -> float:
    pr = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * pr * rec / (pr + rec)


def roc_auc(y_true: pd.Series, y_pred: pd.DataFrame) -> float:
    data = pd.DataFrame({'pred': y_pred, 'true': y_true}).sort_values(by='pred', ascending=False)
    y_sorted = np.array(data['true'])
    y_pred_sorted = np.array(data['pred'])
    result = 0
    cl_1idx = np.where(y_sorted == 1)[0]
    for i in cl_1idx:
        result += np.sum((y_sorted[i:] == 0) & (y_pred_sorted[i:] != y_pred_sorted[i]))
        result += np.sum((y_sorted[i:] == 0) & (y_pred_sorted[i:] == y_pred_sorted[i])) / 2
    return result / (np.sum(y == 1) * np.sum(y == 0))


def recall(y_true: pd.Series, y_pred: pd.DataFrame) -> float:
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    return tp / (tp + fn)


def precision(y_true: pd.Series, y_pred: pd.DataFrame) -> float:
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return tp / (tp + fp)


def accuracy(y_true: pd.Series, y_pred: pd.DataFrame) -> float:
    return (y_true == y_pred).mean()


X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

cls = MyLogReg(n_iter=100, learning_rate=0.03)
cls.fit(X_train, y_train, verbose=True)

y_hat = cls.predict(X_test)

# X, y = make_regression(
# n_samples=400,
# n_features=14,
# n_informative=5,
# noise=15)
# X = pd.DataFrame(X)
# y = pd.Series(y)
# #X, _, y, _ = train_test_split(X, y, test_size=0.2, random_state=42)
#
# line = MyLogReg(n_iter=50, learning_rate=0.1,metric='precision')
# line.fit(X, y, verbose=True)
# #X, y = make_regression(n_samples=100, n_features=14, n_informative=10, noise=15, random_state=66)
# # X = pd.DataFrame(X)
# # y = pd.Series(y)
# #X.columns = [f'col_{col}' for col in X.columns]
# print(line.get_best_score())
# print('SUM:--->  ',line.get_coef().sum())
