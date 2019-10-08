from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from pathlib import Path
import Preprocessing

FILEPATH = Path.cwd() / "reddit-comment-classification-comp-551" / "reddit_train.csv"

if __name__ == "__main__":
    array = Preprocessing.process()
    X = array[0]
    y = array[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    clf = MultinomialNB().fit(X_train, y_train)
    print(clf.predict(count_vect.transform(["I think Urgot is more appealing.I mean there's even a sideboob on his splash art. Full naked with nothing to imagine isn't as sexy."])))
