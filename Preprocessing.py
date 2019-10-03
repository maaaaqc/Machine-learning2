from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from pathlib import Path
import numpy as np
import csv
import spacy

nlp = spacy.load("en", disable=['parser', 'ner'])
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


def lemmatize_all():
    train_set = read_csv(FILEPATH)
    train_x = train_set[:, 0]
    train_y = train_set[:, 1]
    # for i in range(train_x.shape[0]):
    #     train_x[i] = np.str_(" ".join([token.lemma_ for token in nlp(str(train_x[i]))]))
    vectorizer = CountVectorizer(min_df=1, ngram_range=(1, 1))
    vectorizer.fit_transform(train_x)
    feature_name = vectorizer.get_feature_names()
    print(feature_name)


if __name__ == "__main__":
    lemmatize_all()
