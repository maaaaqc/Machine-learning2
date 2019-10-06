import Preprocessing
import numpy as np

class NaiveBayes:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def fit_bernouli(self, i):
        count = 0
        if self.y == i:
            count += 1

    # theta j,k
    def count_occurrance(self, j, k):
        # (# of examples where xj = 1 and y = k)/(# of examples where y = k)
        sum1 = 1
        sum2 = 2
        for i in range(self.x.shape[0]):
            if self.y[i] == k:
                sum2 = sum2 + 1
                if self.x[i][j] == 1:
                    sum1 = sum1 + 1
        return sum1 / sum2
        pass

    def predict(self, x_test):
        max_probability = 0
        max_probability_index = -1
        for i in range(self.y):
            # k = y[i]
            t1 = np.math.factorial(np.sum(x_test))
            t2 = np.prod(x_test)
            total = 1
            for j in range(x_test.shape()[1]):
                total *= np.math.pow(self.count_occurrance(j, self.y[i]), x_test[j])
            if (t1 / t2 * total > max_probability):
                max_probability = t1 / t2 * total
                max_probability_index = self.y[i]
        return max_probability_index
