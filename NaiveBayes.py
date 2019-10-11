import numpy as np
import Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import Evaluation


class NaiveBayes(object):
    def __init__(self):
        pass

    def fit(self, x, y):
        x = x.toarray()
        numOfSamples = np.shape(x)[0]
        numOfFeatures = np.shape(x)[1]
        x = (x > 0).astype(np.int_)

        self.subclasses = ['9', '13', '10', '6', '11', '0', '15', '17', '8', '14', '1', '7', '18', '4', '19', '5', '3', '12', '2', '16']
        self.weightsX = np.zeros((len(self.subclasses), numOfFeatures))
        self.weightsY = np.zeros((len(self.subclasses), 1))
        for i in range(numOfSamples):
            for j in range(len(self.subclasses)):
                if y[i] == self.subclasses[j]:
                    self.weightsY[j] += 1
                    self.weightsX[j] += x[i]
                    continue
        # Record log_probabilities in weightsX and Y
        for i in range(len(self.subclasses)):
            for j in range(numOfFeatures):
                self.weightsX[i][j] = (self.weightsX[i][j] + 1) / (self.weightsY[i] + 2)
        for i in range(len(self.subclasses)):
            self.weightsY[i] = (self.weightsY[i] + 1) / (numOfSamples + 2)
        return

    def predict(self, x_test):
        x_test = x_test.toarray()
        x_test = (x_test > 0).astype(np.int_)
        numOfSamples = np.shape(x_test)[0]
        numOfFeatures = np.shape(x_test)[1]
        resultY = []
        for i in range(numOfSamples):
            probs = []
            for n in range(len(self.subclasses)):
                logProbability = 1
                for j in range(numOfFeatures):
                    if x_test[i][j] == 1:
                        logProbability = logProbability * self.weightsX[n][j]
                    else:
                        logProbability = logProbability * (1 - self.weightsX[n][j])
                logProbability = logProbability * self.weightsY[n]
                probs.append(logProbability)
            max_log_probability = max(probs)
            for m in range(len(self.subclasses)):
                if probs[m] == max_log_probability:
                    resultY.append(self.subclasses[m])
                    break
        return np.asarray(resultY)

if __name__ == "__main__":
    print("BerNb")
    mnb = NaiveBayes()
    train_data = Preprocessing.process_train()

    x_all = train_data[:, 0]
    y_all = train_data[:, 1]
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=0)
    vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1, 1), stop_words='english', strip_accents='ascii')
    output = vectorizer.fit_transform(x_train)
    x_train = output[:, :]
    mnb.fit(x_train, y_train)
    x_test = vectorizer.transform(x_test)[:, :]
    y_pred = mnb.predict(x_test)
    print("pre:", y_pred)
    print("test:", y_test)
    print(Evaluation.evaluate(y_pred, y_test))