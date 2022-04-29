import numpy
import numpy as np

from math import exp
from math import log2
from numpy.linalg import inv
class MyLinearUnivariateRegression:
    def __init__(self):
        self.intercept_ = 0.0
        self.coef_ = []

    def fit(self, x, y):
        maty = numpy.matrix(y)
        mat = numpy.matrix(x)
        trans = mat.transpose()
        firstMul = numpy.matmul(trans, mat)
        invMul = inv(firstMul)
        secondMul = numpy.matmul(invMul, trans)
        mul = numpy.matmul(secondMul, maty)


        self.intercept_ = mul.item(0, 0)
        self.coef_.append(mul.item(0, 0))
        self.coef_.append(mul.item(1, 0))

    # predict the outputs for some new inputs (by using the learnt model)
    def predict(self, x):
        if (isinstance(x[0], list)):
            return [self.intercept_ + self.coef_[0] * val[0] + self.coef_[1] * val[1] for val in x]
        else:
            return [self.intercept_ + self.coef_ * val for val in x]