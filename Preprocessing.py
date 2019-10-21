from pathlib import Path
import numpy as np
import spacy
import json
import csv
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# lemmatizer = WordNetLemmatizer()
porter = PorterStemmer()
nlp = spacy.load("en", disable=['parser', 'ner', 'tagger'])
TRAINPATH = Path.cwd() / "reddit-comment-classification-comp-551" / "reddit_train.csv"
TESTPATH = Path.cwd() / "reddit-comment-classification-comp-551" / "reddit_test.csv"
CONFIG = Path.cwd() / "config.json"

cjk_pattern = re.compile(u'[\u3300-\u33ff\ufe30-\ufe4f\uf900-\ufaff\U0002f800-\U0002fa1f\u30a0-\u30ff\u2e80-\u2eff\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf]')
han_pattern = re.compile(u'[\u3131-\ucb4c]')
url_pattern = re.compile('((www\.[^\s]+)|(https?://[^\s]+))')


def read_csv(path):
    fn = open(str(path), "r", encoding="utf-8")
    ret = csv.reader(fn, delimiter=',')
    data = []
    for x in ret:
        data.append(x)
    data = np.array(data)
    fn.close()
    data = data[1:, :]
    id_num = data[:, 0]
    data = data[:, 1:]
    return [id_num, data]


def process_train():
    train_set = read_csv(TRAINPATH)[1]
    for i in range(train_set.shape[0]):
        train_set[i, 0] = process_sentence(train_set[i, 0])
    train_set[:, 1] = categorize(train_set[:, 1])
    return train_set


def process_test():
    test_set = read_csv(TESTPATH)
    ids = test_set[0]
    test_set = test_set[1]
    for i in range(test_set.shape[0]):
        test_set[i, 0] = process_sentence(test_set[i, 0])
    return [ids, test_set]


def process_sentence(data):
    # data = clean_url(data)
    data = clean_underscore(data)
    data = clean_repeat(data)
    data = clean_cjk(data)
    data = clean_hangul(data)
    data = lemmatize_spacy(data)
    # data = lemmatize_nltk(data)
    # data = stem_list(data)
    data = clean_number(data)
    return data


def categorize(train_y):
    with open(CONFIG) as json_file:
        target = json.load(json_file)
    for i in range(len(train_y)):
        for key in target:
            if train_y[i].lower() == key.lower():
                train_y[i] = target[key]
    return train_y


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
        if not cjk_pattern.search(token) is None:
            token = "japanese"
        new_tok.append(token)
    data = " ".join(new_tok)
    return data


def clean_hangul(data):
    tokens = data.split(" ")
    new_tok = []
    for token in tokens:
        if not han_pattern.search(token) is None:
            token = "korean"
        new_tok.append(token)
    data = " ".join(new_tok)
    return data


def lemmatize_spacy(data):
    data = np.str_(" ".join([token.lemma_ for token in nlp(str(data))]))
    return data


def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_nltk(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(res_words)


def stem_list(input_list):
    input_list = " ".join([porter.stem(word) for word in input_list.split(" ")])
    return input_list
