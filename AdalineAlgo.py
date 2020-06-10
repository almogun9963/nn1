import numpy as np
# https://yasoob.me/2013/08/07/the-self-variable-in-python-explained/
# this is for the variable self and init
# we chose learning_rate 0.0001, epochs 5000
# https://stackoverflow.com/questions/43559908/disadvantages-of-running-more-epochs-on-same-learn-rate

class AdalineAlgo(object):
    # init self
    def __init__(self, learning_rate=0.0001, epochs=5000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        

    ## function to train and fix weights
    def train(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        costs = np.array([])
        for i in range(self.epochs):
            output = np.dot(X, self.w_[1:]) + self.w_[0]
            errors = (y - output)
            self.w_[1:] += self.learning_rate * np.dot(X.T, errors)
            self.w_[0] += self.learning_rate * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            costs = np.append(costs, [cost])
        return self.w_, costs

    # predict value
    def predict(self, X):
        return np.where(np.dot(X, self.w_[1:]) + self.w_[0] >= 0.0, 1, -1)
