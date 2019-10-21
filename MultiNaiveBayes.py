from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
import Preprocessing
import Evaluation
import CSVChange


class MultiNaiveBayes:
    def __init__(self):
        pass

    def k_fold(self, k, min, max, alp):
        mnb = MultinomialNB(alpha=alp)
        train_data = Preprocessing.process_train()
        np.random.shuffle(train_data)
        groups = np.array_split(train_data, k, axis=0)
        acc = 0
        for i in range(k):
            val_set = groups[i][:, :-1]
            true_val = groups[i][:, -1]
            train_set = np.concatenate(groups[:i] + groups[i+1:], axis=0)
            x_train = train_set[:, :-1]
            y_train = train_set[:, -1]
            vectorizer = TfidfVectorizer(min_df=min, max_df=max, ngram_range=(1, 1), stop_words=None, max_features=50000, analyzer='word')
            output = vectorizer.fit_transform(x_train)
            x_train = output[:, :]
            mnb.fit(x_train, y_train)
            val_set = vectorizer.transform(val_set)[:, :]
            y_pred = mnb.predict(val_set)
            acc += Evaluation.evaluate(y_pred, true_val)
        acc /= k
        return acc

    def predict_and_test(self, min, max, alp):
        mnb = MultinomialNB(alpha=alp)
        train_data = Preprocessing.process_train()
        np.random.shuffle(train_data)
        x_all = train_data[:, 0]
        y_all = train_data[:, 1]
        x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=0)
        vectorizer = TfidfVectorizer(min_df=min, max_df=max, ngram_range=(1, 1), stop_words=None, max_features=50000, analyzer='word')
        output = vectorizer.fit_transform(x_train)
        x_train = output[:, :]
        mnb.fit(x_train, y_train)
        x_test = vectorizer.transform(x_test)[:, :]
        y_pred = mnb.predict(x_test)
        return Evaluation.evaluate(y_pred, y_test)

    def predict_and_write(self, min, max, alp):
        mnb = MultinomialNB(alpha=alp)
        train_data = Preprocessing.process_train()
        test_set = Preprocessing.process_test()
        x_test = test_set[1]
        id_data = test_set[0]
        x_all = train_data[:, 0]
        y_all = train_data[:, 1]
        vectorizer = TfidfVectorizer(min_df=min, max_df=max, ngram_range=(1, 1), stop_words='english', strip_accents='ascii')
        output = vectorizer.fit_transform(x_all)
        x_all = output[:, :]
        mnb.fit(x_all, y_all)
        x_test = vectorizer.transform(x_test.ravel())[:, :]
        y_pred = mnb.predict(x_test).ravel()
        y_pred = y_pred.reshape(len(y_pred), 1)
        y_pred = np.concatenate((id_data.reshape(len(id_data), 1), y_pred), axis=1)
        first = ["Id", "Category"]
        y_pred = np.concatenate((np.array(first).reshape(1, 2), y_pred), axis=0)
        np.savetxt("prediction.csv", y_pred, fmt="%s", delimiter=",")
        CSVChange.write()
        return
