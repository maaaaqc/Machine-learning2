from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from pathlib import Path
import numpy as np
import Preprocessing
import Evaluation

FILEPATH = Path.cwd() / "reddit-comment-classification-comp-551" / "reddit_train.csv"


class MultiNaiveBayes:
    def count_vectorize_all(self, train_x):
        vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1), stop_words='english', strip_accents='ascii')
        output = vectorizer.fit_transform(train_x)
        return output

    def tfidf_vectorize_all(self, train_x):
        vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1), stop_words='english', strip_accents='ascii')
        output = vectorizer.fit_transform(train_x)
        return output


if __name__ == "__main__":
    mnb = MultinomialNB()
    train_data = Preprocessing.process_train()
    np.random.shuffle(train_data)
    x_all = train_data[:, 0]
    y_all = train_data[:, 1]
    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=0)
    vectorizer = TfidfVectorizer(min_df=3, ngram_range=(1, 1), stop_words='english', strip_accents='ascii')
    output = vectorizer.fit_transform(x_train)
    x_train = output[:, :]
    naiveBayes = mnb.fit(x_train, y_train)
    x_test = vectorizer.transform(x_test)[:, :]
    y_pred = mnb.predict(x_test)
    print(Evaluation.evaluate(y_pred, y_test))