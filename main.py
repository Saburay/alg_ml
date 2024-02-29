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
    def __init__(self, n_iter=100, learning_rate=0.1, weights=None):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

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

        ones = 1

        n = len(self.y)
        x.insert(loc=0, column='ones', value=ones)#дополняем переданную матрицу фичей x единичным столбцом слева.
        #print(f'self_X_ones:\n{x}')
        v_weights = np.ones(x.shape[1])#Определить сколько фичей передано и создать вектор весов,
                               # состоящий из одних единиц соответствующей длинны: т.е. количество фичей + 1.
        #print(f'self_v_wights:{len(v_weights)}{v_weights}')
        s = 'start'
        for i in range(self.n_iter):
            pred_y = x.dot(v_weights)
            #print(f'pred_y:{pred_y}\n{type(pred_y)},self_y {type(self.y)})')
            mse = sum((pred_y-self.y)**2)/n
            #print(f'mse{mse}')
            pred_min_y =(pred_y - self.y)
            #print(f'pred_min_y:    {pred_min_y}')
            gr = ((2/n)*((pred_y-self.y).dot(x)))
            #print(f'gr------->{gr}')
            v_weights = v_weights-self.learning_rate*gr
            self.weights = v_weights
            if verbose:
                if i == 1 or i % 10 == 0:
                    print(f'{s}|{mse}')
                    s = i
    def get_coef(self):
        return self.weights[1:]

X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
X = pd.DataFrame(X)
y = pd.Series(y)
X.columns = [f'col_{col}' for col in X.columns]

#print(X)

zz = MyLineReg()
zz.fit(X,y,verbose=True)

#------------------------------------------------------
# import numpy as np
# import pandas as pd
# class MyLineReg():
#     '''
#     n_iter — количество шагов градиентного спуска.
#     По-умолчанию: 100
#     learning_rate — коэффициент скорости обучения градиентного спуска.
#     По-умолчанию: 0.1
#     weights — хранит веса модели
#     '''
#     def __init__(self, n_iter=100, learning_rate=0.1, weights=None):
#         self.n_iter = n_iter
#         self.learning_rate = learning_rate
#         self.weights = weights
#
#     def __repr__(self):
#         return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'
#
#     def fit(self, x, y, verbose=False):
#         '''
#         -:param x: — все фичи в виде датафрейма пандаса.
#                Примечание: даже если фича будет всего одна это все равно будет датафрейм, а не серия.
#         -:param y: — целевая переменная в виде пандасовской серии.
#         -:param verbose: — указывает на какой итерации выводить лог.
#                Например, значение 10 означает, что на каждой 10 итерации градиентного спуска будет печататься лог.
#                Значение по умолчанию: False.
#         '''
#         self.x = pd.DataFrame(x)
#         self.y = y = pd.Series(y)
#         self.verbose = verbose
#
#         ones = np.ones(len(x))
#         #ones = x.shape[0]
#         n = len(self.y)
#         x.insert(loc=0, column='ones', value=ones)#дополняем переданную матрицу фичей x единичным столбцом слева.
#         v_weights = np.ones(x.shape[1])#Определить сколько фичей передано и создать вектор весов,
#                                # состоящий из одних единиц соответствующей длинны: т.е. количество фичей + 1.
#         for i in range(self.n_iter):
#             s = 'start'
#             pred_y = self.x*v_weights
#             mse = sum((pred_y-self.y)**2)/n
#             gr = ((2/n)*((pred_y-self.y)*self.x))
#             v_weights = v_weights-self.learning_rate*gr
#             self.weights = v_weights
#             if verbose:
#                 if i == 1 or i % 10 == 0:
#                     print(f'{s}|{mse}')
#                     s = i
#     def get_coef(self):
#         return self.weights[1:]
#
#
#
#
# #y = np.ones([5,1])
# print()
