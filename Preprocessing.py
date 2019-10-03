from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from pathlib import Path
import numpy as np
import csv
import spacy


nlp = spacy.load("en", disable=['parser', 'ner', 'tagger'])
FILEPATH = Path.cwd() / "reddit-comment-classification-comp-551" / "reddit_train.csv"


def read_csv(path):
    fn = open(str(path), "r")
    ret = csv.reader(fn, delimiter=',')
    data = []
    for x in ret:
        data.append(x)
    data = np.array(data)
    fn.close()
    # delete header
    data = data[1:, :]
    # delete id column
    data = data[:, 1:]
    return data


def vectorize_all():
    train_set = read_csv(FILEPATH)
    train_x = train_set[:, 0]
    train_y = train_set[:, 1]
    train_x = lemmatize_all(train_x)
    vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1))
    vectorizer.fit_transform(train_x)
    feature_name = vectorizer.get_feature_names()


def lemmatize_all(data):
    for i in range(data.shape[0]):
        data[i] = np.str_(" ".join([token.lemma_ for token in nlp(str(data[i]))]))
    return data


if __name__ == "__main__":
    vectorize_all()
