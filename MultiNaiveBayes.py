from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import Preprocessing
import Evaluation
import json

CONFIG = Path.cwd() / "config.json"


class MultiNaiveBayes:
    def count_vectorize_all(self, train_x):
        vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1), stop_words='english', strip_accents='ascii')
        output = vectorizer.fit_transform(train_x)
        return output

    def tfidf_vectorize_all(self, train_x):
        vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1), stop_words='english', strip_accents='ascii')
        output = vectorizer.fit_transform(train_x)
        return output

    def predict(self):
        mnb = MultinomialNB()
        train_data = Preprocessing.process_train()
        test_set = Preprocessing.process_test()
        x_test = test_set[1]
        id_data = test_set[0]
        x_all = train_data[:, 0]
        y_all = train_data[:, 1]
        vectorizer = TfidfVectorizer(min_df=3, max_df=0.3, ngram_range=(1, 1), stop_words='english', strip_accents='ascii')
        output = vectorizer.fit_transform(x_all)
        x_all = output[:, :]
        mnb.fit(x_all, y_all)
        x_test = vectorizer.transform(x_test.ravel())[:, :]
        y_pred = mnb.predict(x_test).ravel()
        with open(CONFIG) as json_file:
            target = json.load(json_file)
        #for i in range(y_pred.shape[0]):
        #    for key in target:
        #        if target[key] == int(y_pred[i]):
        #            print("Hi")
        #            y_pred[i] = key
        y_pred = y_pred.reshape(len(y_pred), 1)
        y_pred = np.concatenate((id_data.reshape(len(id_data), 1), y_pred), axis=1)
        first = ["Id", "Category"]
        y_pred = np.concatenate((np.array(first).reshape(1, 2), y_pred), axis=0)
        np.savetxt("prediction.csv", y_pred, fmt="%s", delimiter=",")


if __name__ == "__main__":
    mnb = MultinomialNB()
    train_data = Preprocessing.process_train()
    np.random.shuffle(train_data)
    x_all = train_data[:, 0]
    y_all = train_data[:, 1]
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=0)
    vectorizer = TfidfVectorizer(min_df=3, max_df=0.3, ngram_range=(1, 1), stop_words='english', strip_accents='ascii')
    output = vectorizer.fit_transform(x_train)
    x_train = output[:, :]
    naiveBayes = mnb.fit(x_train, y_train)
    x_test = vectorizer.transform(x_test)[:, :]
    y_pred = mnb.predict(x_test)
    print(Evaluation.evaluate(y_pred, y_test))
