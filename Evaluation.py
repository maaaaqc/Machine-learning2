import MultiNaiveBayes


def evaluate(y_true, y_label):
    total = 0
    # y_true = y_true.reshape(y_true.shape[0], 1)
    for i in range(y_true.shape[0]):
        if y_true[i] == y_label[i]:
            total = total + 1
    return total / y_true.shape[0]


if __name__ == "__main__":
    s = MultiNaiveBayes.MultiNaiveBayes()
    s.predict()
