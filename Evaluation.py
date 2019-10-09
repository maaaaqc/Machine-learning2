from pathlib import Path
import numpy as np
import Preprocessing
import Evaluation

FILEPATH = Path.cwd() / "reddit-comment-classification-comp-551" / "reddit_test.csv"


def evaluate(y_true, y_label):
    total = 0
    y_true = y_true.reshape(y_true.shape[0], 1)
    for i in range(y_true.shape[0]):
        if y_true[i] == y_label[i]:
            total = total + 1
    return total / y_true.shape[0]
