import MultiNaiveBayes
import NaiveBayes
import numpy


def evaluate(y_true, y_label):
    total = 0
    for i in range(y_true.shape[0]):
        if y_true[i] == y_label[i]:
            total = total + 1
    return total / y_true.shape[0]


def test():
    mb = MultiNaiveBayes.MultiNaiveBayes()
    mb.predict_and_test(1, 0.3, 0.167)


def generate_result():
    mb = MultiNaiveBayes.MultiNaiveBayes()
    mb.predict_and_write(1, 0.3, 0.167)


if __name__ == "__main__":
    generate_result()
