import numpy as np
import spacy
nlp = spacy.load("en", disable=['parser', 'ner'])


if __name__ == "__main__":
    sentence = "The striped bats are hanging on their feet for best"
    sol = " ".join([token.lemma_ for token in nlp(sentence)])
    print(sol)
