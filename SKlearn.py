from sklearn.naive_bayes import BernoulliNB
from pathlib import Path
import Preprocessing
import NaiveBayes
import numpy as np



FILEPATH = Path.cwd() / "reddit-comment-classification-comp-551" / "reddit_train.csv"

def k_fold(data, k):
    np.random.shuffle(data)
    data_subsets = np.array_split(data, k, axis = 0)
    avg_acc = 0
    # start = time.time()

    for i in range(k):
        training_data = np.concatenate(data_subsets[:i] + data_subsets[i + 1:], axis = 0)
        validation_data = data_subsets[i]
        gnb = BernoulliNB()
        gnb.fit(training_data[:, 0:-1], training_data[:, -1])
        y_perdict = gnb.predict(validation_data[:, 0:-1])
        avg_acc = avg_acc + evaluate_acc(validation_data[:, -1].reshape(validation_data.shape[0], 1), y_perdict)
    return

def evaluate_acc(Y_true_label, Y_target_label):
    total = 0
    for i in range(Y_true_label.shape[0]):
        if Y_true_label[i] == Y_target_label[i]:
            total = total + 1
    return total / Y_true_label.shape[0]

if __name__ == "__main__":
    print("SKlearn")
    gnb = BernoulliNB()
    train_data = Preprocessing.process(FILEPATH)


    train_data[1] = train_data[1].reshape(train_data[1].shape[0], 1)
    # print(train_data[1].shape)
    train_data_x = (train_data[0])[:, :]
    train_data_y = (train_data[1])[:, :]
    # training_data = np.concatenate((train_data_x, train_data_y), axis=1)
    # training_data = np.append(train_data_x, train_data_y, axis=1)
    print(train_data_x.shape)
    print(train_data_y.shape)
    # print(training_data.shape)
    # # np.random.shuffle(training_data)
    #
    # train_data_x = (training_data)[:, 0:-1]
    # train_data_y = (training_data)[:, -1]


    # print((train_data[1])[0:200, :])
    naiveBayes = gnb.fit(train_data_x[10001:, :], train_data_y[10001:, :])
    y_pred = gnb.predict(train_data_x[0:10000, :])
    #
    # # print(y_pred)
    # # print((train_data[1])[201:220, :])
    # nb = NaiveBayes.NaiveBayes()
    y_true = train_data_y[0:10000, :]
    # print(nb.evaluate(y_pred, y_true))

    # print(y_pred)
    # print(y_true)

    # training_data = np.concatenate((train_data_x, train_data_y), axis = 1)
    n = NaiveBayes.NaiveBayes()
    print(n.evaluate(y_pred, y_true))
    # print(training_data.shape)
    # k_fold(training_data, 5)

