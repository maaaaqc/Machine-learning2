from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import numpy as np
import spacy
import json
import csv
import re
import NaiveBayes
import BerNB
import NBLJW

nlp = spacy.load("en", disable=['parser', 'ner', 'tagger'])
FILEPATH = Path.cwd() / "reddit-comment-classification-comp-551" / "reddit_train.csv"
CONFIG = Path.cwd() / "config.json"

cjk_pattern = re.compile(u'[\u3300-\u33ff\ufe30-\ufe4f\uf900-\ufaff\U0002f800-\U0002fa1f\u30a0-\u30ff\u2e80-\u2eff\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf]')
han_pattern = re.compile(u'[\u3131-\ucb4c]')
url_pattern = re.compile('((www\.[^\s]+)|(https?://[^\s]+))')


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


def process(file_path):
    train_set = read_csv(file_path)
    # train_set = train_set[0:10000, :]
    train_x = train_set[:, 0]
    train_y = train_set[:, 1]
    for i in range(train_x.shape[0]):
        train_x[i] = clean_url(train_x[i])
        train_x[i] = clean_underscore(train_x[i])
        train_x[i] = clean_repeat(train_x[i])
        train_x[i] = clean_cjk(train_x[i])
        train_x[i] = clean_hangul(train_x[i])
        train_x[i] = lemmatize_all(train_x[i])
        train_x[i] = clean_number(train_x[i])
    # train_x = count_vectorize_all(train_x)
    train_x = tfidf_vectorize_all(train_x)
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
    vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1), stop_words='english', strip_accents='ascii')
    output = vectorizer.fit_transform(train_x)
    # np.savetxt("feature.txt", vectorizer.get_feature_names(), fmt="%s")
    return output


def tfidf_vectorize_all(train_x):
    vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1), stop_words='english', strip_accents='ascii')
    output = vectorizer.fit_transform(train_x)
    return output


def clean_url(data):
    data = url_pattern.sub('url', data)
    return data


def clean_underscore(data):
    data = np.str_(" ".join(str(data).split("_")))
    return data


def clean_number(data):
    tokens = data.split(" ")
    new_tok = []
    for token in tokens:
        if not token.isnumeric():
            new_tok.append(token)
    data = " ".join(new_tok)
    return data


def clean_repeat(data):
    tokens = data.split(" ")
    new_tok = []
    for token in tokens:
        token = re.sub(r'(.)\1\1+', r'\1', token)
        new_tok.append(token)
    data = " ".join(new_tok)
    return data


def clean_cjk(data):
    tokens = data.split(" ")
    new_tok = []
    for token in tokens:
        if not cjk_pattern.search(token) == None:
            token = "japanese"
        new_tok.append(token)
    data = " ".join(new_tok)
    return data


def clean_hangul(data):
    tokens = data.split(" ")
    new_tok = []
    for token in tokens:
        if not han_pattern.search(token) == None:
            token = "korean"
        new_tok.append(token)
    data = " ".join(new_tok)
    return data


def lemmatize_all(data):
    data = np.str_(" ".join([token.lemma_ for token in nlp(str(data))]))
    return data


if __name__ == "__main__":
    train_data = process(FILEPATH)
    # test_data = process(Path.cwd() / "reddit-comment-classification-comp-551" / "reddit_test.csv")
    print(type(train_data[0]))
    print(type(train_data[1]))

    train_data[1] = train_data[1].reshape(train_data[1].shape[0], 1)

    train_data_x = (train_data[0])[:, :]
    train_data_y = (train_data[1])[:, :]

    naiveBayes = NaiveBayes.NaiveBayes()

    # naiveBayesLJW = NBLJW.NB(train_data_x, train_data_y)
    # y_target = naiveBayesLJW.predict(train_data_x[51:70, :])
    # y_true = train_data_y[51:70, :]
    # print("true", y_true)
    # print("y_target", y_target)
    #
    # # naiveBayesLJW.evaluate(y_target, y_true)
    #
    #
    # # print(train_data_x.shape)
    # # # print((train_data[0])[201:300,:])
    naiveBayes.fit(train_data_x, train_data_y)
    y_target = naiveBayes.predict((train_data_x[51:70, :]))
    y_true = train_data_y[51:70, :]
    naiveBayes.predict(y_target, y_true)
    #
    # # print("evaluate:", .evaluate(y_true, y_target))
    #
    # # nbshx = BerNB.BerNB(1)
    # # nbshx.fit((train_data[0])[0:2000, :], (train_data[1])[0:2000,:])
    # # y_target = nbshx.predict((train_data[0])[51:70, 0:50])
    # # y_true = (train_data[1])[51:70, :]
    # # print(y_target)
    # # print(y_true)
    #
    # # naiveBayes.fit(train_data_x[:, :], train_data_y[:, :])
    # # y_target = naiveBayes.predict(train_data_x[51:70, :])
    # # y_true = train_data_y[51:70, :]
    # # naiveBayes.evaluate(y_target, y_true)
    #
    # # for i in range(train_data[1].shape[0]):
    # #     print(i, naiveBayes.predict((train_data[0])[i]), (train_data[1][i]))

