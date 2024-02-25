import numpy as np
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
        self.weighs = weights

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
        self.x = x
        self.y = y
        self.verbose = verbose

        ones = np.ones(len(x))
        #ones = x.shape[0]
        x.insert(loc=0, column='ones', value=ones)#дополняем переданную матрицу фичей x единичным столбцом слева.
        w = np.ones(x.shape[1])#Определить сколько фичей передано и создать вектор весов,
                           # состоящий из одних единиц соответствующей длинны: т.е. количество фичей + 1.


y = np.ones([5,1])
print(y)
