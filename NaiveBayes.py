import numpy as np


class NaiveBayes:
    def __init__(self):
        pass

    def fit_bernouli(self, i):
        count = 0
        if self.y == i:
            count += 1

    def count_occurrance(self):
        for i in range(self.x.shape[0]):
            self.occurrance_table[self.y_dic[self.y[i][0]]][-1] += 1
            for j in range(self.x.shape[1]):
                if self.x[i, j] != 0:
                    self.occurrance_table[self.y_dic[self.y[i][0]]][j] += 1

    def theta(self, j, k):
        return (self.occurrance_table[self.y_dic[k]][j] + 1) / (self.occurrance_table[self.y_dic[k]][-1] + 2)

    def fit(self, x, y):
        self.x = x
        self.y = y

        # construct y dictionary
        y_set = set()
        for y in self.y:
            y_set.add(y[0])
        self.y_dic = dict()
        i = 0
        for y in y_set:
            self.y_dic[y] = i
            i += 1
        # print("dic:", self.y_dic)
        # last column is for the occurance of y
        self.occurrance_table = np.full((len(self.y_dic), self.x.shape[1] + 1), 0)
        self.count_occurrance()
        print(self.occurrance_table)


    def predict(self, x_test):
        # x_test = x_test.reshape(x_test.shape[0], 1)
        print("x_test shape:", self.x.shape)
        y_target = np.full((x_test.shape[0], 1), '')

        for i in range(x_test.shape[0]):
            max_probability = -1
            max_probability_index = -1
            for k in self.y_dic.keys():
                total = 0
                for j in range(x_test.shape[1]):
                    theta = self.theta(j, k)
                    total += x_test[i][j] * np.math.log(theta) + (1 - x_test[i][j]) * np.math.log(1 - theta)
                total += np.math.log(self.occurrance_table[self.y_dic[k]][-1] / self.x.shape[0])
                if np.math.exp(-total) > max_probability:
                    max_probability = np.math.exp(-total)
                    max_probability_index = k
            y_target[i] = max_probability_index
        return y_target
