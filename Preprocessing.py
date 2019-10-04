from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import numpy as np
import spacy
import json
import csv
import re


nlp = spacy.load("en", disable=['parser', 'ner', 'tagger'])
FILEPATH = Path.cwd() / "reddit-comment-classification-comp-551" / "reddit_train.csv"
CONFIG = Path.cwd() / "config.json"


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


def process():
    train_set = read_csv(FILEPATH)
    train_x = train_set[:, 0]
    train_y = train_set[:, 1]
    for i in range(train_x.shape[0]):
        train_x[i] = clean_url(train_x[i])
        train_x[i] = clean_underscore(train_x[i])
        train_x[i] = lemmatize_all(train_x[i])
    train_x = count_vectorize_all(train_x)
    train_y = categorize(train_y)
    return [train_x, train_y]


def categorize(train_y):
    with open(CONFIG) as json_file:
        target = json.load(json_file)
    for i in range(len(train_y)):
        for key in target:
            if train_y[i].lower() == key:
                train_y[i] = target[key]
    return train_y


def count_vectorize_all(train_x):
    vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1), stop_words='english')
    output = vectorizer.fit_transform(train_x).toarray()
    np.savetxt("feature.txt", vectorizer.get_feature_names(), fmt="%s")
    return output


def tfidf_vectorize_all(train_x):
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1), stop_words='english')
    output = vectorizer.fit_transform(train_x).toarray()
    return output


def clean_url(data):
    data = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'url', data)
    return data


def clean_underscore(data):
    data = np.str_(" ".join(str(data).split("_")))
    return data


def lemmatize_all(data):
    data = np.str_(" ".join([token.lemma_ for token in nlp(str(data))]))
    return data


if __name__ == "__main__":
    process()
