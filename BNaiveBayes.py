import numpy as np
import csv
def readFromCSV(path):
    fn = open(path, "r")
    ret = csv.reader(fn, delimiter=',')
    data = []
    print(type(ret))
    for x in ret:
        data.append(x)
    data = np.array(data)
    fn.close()
    # delete header
    data = data[1:, :]
    # delete id column
    data = data[:, 1:]
    return data
train = readFromCSV('reddit-comment-classification-comp-551/reddit_train.csv')
# np.savetxt("train.txt", train)
print(train.shape)
print(train[:, -1])
# print("test", readFromCSV('reddit-comment-classification-comp-551/reddit_test.csv'))