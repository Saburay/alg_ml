import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


class MyLineReg():
    '''
    n_iter — количество шагов градиентного спуска.
    По-умолчанию: 100
    learning_rate — коэффициент скорости обучения градиентного спуска.
    weights — хранит веса модели
    metric, который будет принимать одно из следующих значений:
    - mae
    - mse
    - rmse
    - mape
    - r2
    По умолчанию: None
    reg – принимает одно из трех значений: l1, l2, elasticnet
    По умолчанию: None
    l1_coef – принимает значения от 0.0 до 1.0
    По умолчанию: 0
    l2_coef – принимает значения от 0.0 до 1.0
    По умолчанию: 0
    '''
    def __init__(self,  learning_rate, n_iter=100, weights=None, metric=None, reg=None,
                 l1_coef=0, l2_coef=0):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        # self.learning_rate = 0.1
        self.weights = weights
        self.metric = metric
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.reg = reg
        # self.score_dict = {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}

    def __repr__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def fit(self, x, y, verbose=False):
        '''
        -:param x: — все фичи в виде датафрейма пандаса.
               Примечание: даже если фича будет всего одна это все равно будет датафрейм, а не серия.
        -:param y: — целевая переменная в виде пандасовской серии.
        -:param verbose: — указывает на какой итерации выводить лог.
               Например, значение 10 означает, что на каждой 10 итерации градиентного спуска будет печататься лог.
               Значение по умолчанию: False.
        '''
        self.x = pd.DataFrame(x)
        self.y = pd.Series(y)
        self.verbose = verbose

        self.mean_y = y.mean(axis=0)  # среднее значение целевой переменной
        self.best_score = None

        # n = len(self.y)
        n = x.shape[0]
        x.insert(loc=0, column='ones', value=1)  # дополняем переданную матрицу фичей x единичным столбцом слева.
        self.weights = np.ones(x.shape[1])  # Определить сколько фичей передано и создать вектор весов,
        # состоящий из одних единиц соответствующей длинны: т.е. количество фичей + 1.

        s = 'start'
        #use_learning_rate = self.learning_rate
        for iter in range(1, self.n_iter+1):
            use_learning_rate = self.learning_rate
            pred_y = x.dot(self.weights)
            lasso_mse = self.l1_coef * (sum(abs(self.weights)))  # L1 слагаемое к mse
            ridge_mse = self.l2_coef * (sum((self.weights) ** 2))  # L2 слагаемое к mse

            # gr_lasso_mse = (self.l1_coef*(self.weights/abs(self.weights))) #L1 слагаемое к градиенту gr
            gr_lasso_mse = (self.l1_coef * np.sign(self.weights))  # L1 слагаемое к градиенту gr
            gr_ridge_mse = (self.l2_coef * (2 * self.weights))  # L2 слагаемое к градиенту gr

            if self.reg == None:
                mse = sum((self.y - pred_y) ** 2) / n  # функция потерь,метрика mse
                gr = ((2 / n) * ((pred_y - self.y).dot(x)))  # вычисляем градиент
            elif self.reg == 'l1':
                mse = sum((self.y - pred_y) ** 2) / n + lasso_mse
                gr = ((2 / n) * ((pred_y - self.y).dot(x))) + gr_lasso_mse
            elif self.reg == 'l2':
                mse = sum((self.y - pred_y) ** 2) / n + ridge_mse
                gr = ((2 / n) * ((pred_y - self.y).dot(x))) + gr_ridge_mse
            elif self.reg == 'elasticnet':
                mse = sum((self.y - pred_y) ** 2) / n + lasso_mse + ridge_mse
                gr = ((2 / n) * ((pred_y - self.y).dot(x))) + gr_lasso_mse + gr_ridge_mse
            #print(f'iter{iter}')
            try:
                use_learning_rate = self.learning_rate(iter)
            except TypeError:
                pass
            else:
                pass
                #self.learning_rate = self.learning_rate
            #print(f'self.learning_rate: {use_learning_rate}, ####self.weights: {self.weights}  ')
            self.weights = self.weights - use_learning_rate * gr  # шаг размером learning rate в противоположную от градиента сторону

            mae = sum(abs(self.y - pred_y)) / n  # метрика mae
            rmse = (sum((self.y - pred_y) ** 2) / n) ** 0.5  # метрика rmse
            r2 = 1 - (sum((self.y - pred_y) ** 2)) / (sum((self.y - self.mean_y) ** 2))  # метрика r2
            mape = (100 * (sum(abs((self.y - pred_y) / self.y))) / n)  # метрика mape
            score_dict = {'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}  # словарь метрик
            if self.metric != None:
                self.best_score = score_dict[(self.metric).lower()]
            if verbose:
                if iter == 1 or iter % 10 == 0:
                    if self.metric == None:
                        print(f'{s}|loss:{mse}|')
                    else:
                        # self.best_score = score_dict[(self.metric).lower()]
                        print(f'{s}|loss:{mse}|<{self.metric}>:{self.best_score}')
                    s = iter

    def get_coef(self):
        return self.weights[1:]

    def predict(self, xx):
        '''
        -:param x: На вход принимается матрица фичей в виде датафрейма пандаса.
        Возвращается вектор предсказаний
        '''
        self.xx = pd.DataFrame(xx)

        xx.insert(loc=0, column='ones', value=1)  # дополняем переданную матрицу фичей x единичным столбцом слева.
        pred = xx.dot(self.weights)
        return pred

    def get_best_score(self):
        return self.best_score

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

#print(X)

zz = MyLineReg(learning_rate=lambda x: 0.5 * (0.85 ** x),metric='mse')
zz.fit(X,y,verbose=True)
#print(zz.get_coef())
#print(zz.__repr__())
#print(round(zz.get_best_score(),10))
#print(zz.get_best_score())
reg = None
line = MyLineReg(
        n_iter = 50,
        learning_rate = lambda iter: 0.5 * (0.85 ** iter)
)
print(line)

