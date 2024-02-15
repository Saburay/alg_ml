class MyLineReg():
    '''
    n_iter — количество шагов градиентного спуска.
    По-умолчанию: 100
    learning_rate — коэффициент скорости обучения градиентного спуска.
    По-умолчанию: 0.1
    '''
    def __init__(self, n_iter=100, learning_rate=0.1):
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    def __repr__(self):
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'



