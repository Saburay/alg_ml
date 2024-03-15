import numpy as np
import pandas as pd
from sklearn.datasets import make_regression

class MyLineReg():
    '''
    n_iter — количество шагов градиентного спуска.
    По-умолчанию: 100
    learning_rate — коэффициент скорости обучения градиентного спуска.
    По-умолчанию: 0.1
    weights — хранит веса модели
    '''
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None, metric=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        self.metric = metric

    def __repr__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
        # if self.metric!=None:
        #     return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}, <{self.metric}>:'
        # else:
        #     return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'


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

        self.mean_y = x.mean(axis=1)# !!! погонять  и уточнить объявление до добавления вектора едениц или после
        self.best_score = None

        #n = len(self.y)
        n = x.shape[0]
        x.insert(loc=0, column='ones', value=1)#дополняем переданную матрицу фичей x единичным столбцом слева.
        v_weights = np.ones(x.shape[1])#Определить сколько фичей передано и создать вектор весов,
                               # состоящий из одних единиц соответствующей длинны: т.е. количество фичей + 1.
        s = 'start'
        #best_score = None
        for i in range(self.n_iter):
            pred_y = x.dot(v_weights)
            mse = sum((self.y - pred_y) ** 2) / n
            #pred_min_y = (pred_y - self.y)
            gr = ((2/n)*((pred_y-self.y).dot(x)))#Вычисляем градиент
            v_weights = v_weights-self.learning_rate*gr#шаг размером learning rate в противоположную от градиента сторону
            self.weights = v_weights

            mse = sum((self.y - pred_y) ** 2) / n
            mae = sum(self.y - pred_y) / n
            rmse = (sum((self.y - pred_y) ** 2) / n) ** 0.5
            r2 = 1 - (sum((self.y - pred_y) ** 2) / (sum((self.y - self.mean_y) ** 2)))
            mape = (100 / n) * (sum(abs(self.y - pred_y) / self.y))
            score_dict = {'mse':mse,'mae':mae,'rmse':rmse,'r2':r2,'mape':mape}
            if verbose:
                if i == 1 or i % 10 == 0:
                    #print(f'{s}|loss:{mse}|')
                    if self.metric==None:
                        print(f'{s}|loss:{mse}|')
                    else:
                        self.best_score = score_dict[(self.metric).lower()]
                        print(f'{s}|loss:{mse}|<{self.metric}>:{self.best_score}')
                    s = i


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

zz = MyLineReg(metric='mae')
zz.fit(X,y,verbose=True)
#print(zz.get_coef())
#print(zz.__repr__())
print(zz.get_best_score())

