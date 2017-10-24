#! env/bin/python


import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier


__masculine__ = set(['actor', 'author', 'boy', 'brave', 'bachelor', 'bridegroom',
       'brother', 'conductor', 'dad', 'daddy', 'duke', 'emperor', 'father',
       'father-in-law', 'fiance', 'gentleman', 'his', 'him', 'governor',
       'grandfather', 'grandson', 'headmaster', 'heir', 'hero', 'host',
       'hunter', 'husband', 'king', 'lad', 'landlord', 'lord', 'male',
       'man', 'manager', 'masseur', 'milkman', 'monitor', 'monk',
       'murderer', 'nephew', 'papa', 'policeman', 'postman', 'postmaster',
       'priest', 'prince', 'prophet', 'proprietor', 'prosecutor',
       'protector', 'shepherd', 'sir', 'son', 'son-in-law', 'stepfather',
       'stepson', 'steward', 'sultan', 'tailor', 'testator', 'uncle',
       'usher', 'waiter', 'washerman', 'widower', 'wizard'])


__feminine__ = set(['actress', 'spinster', 'girl', 'bride', 'sister', 'conductress',
       'countess', 'czarina', 'mum', 'mummy', 'duchess', 'empress',
       'mother', 'mother-in-law', 'fiancee', 'lady', 'giantess', 'goddess',
       'grandmother', 'granddaughter', 'headmistress', 'heiress',
       'heroine', 'hostess', 'wife', 'queen', 'lass', 'landlady', 'lady',
       'female', 'woman', 'manageress', 'masseuse', 'mistress', 'nun',
       'mrs', 'niece', 'mama', 'policewoman', 'postwoman', 'prietess',
       'princess', 'madam', 'daughter', 'daughter-in-law', 'step-mother',
       'step-daughter', 'stewardess', 'tailoress', 'testatrix', 'aunt',
       'usherette', 'waitress', 'washerwoman', 'widow', 'witch', 'bitch',
       'jenny', 'her', 'hers', 'she', 'maid'])


def data_classes(messages):
    gender = []
    msgs = [re.sub(r"\b\d+\b|![:alpha:]|\d|\W|_|\b\w{1,2}\b", " ", m).lower() for m in messages]
    for msg in msgs:
        msg_bag = msg.split(' ')
        if any([True if w in __feminine__ else False for w in msg_bag]):
            gender.append(1) # F
        elif any([True if w in __masculine__ else False for w in msg_bag]):
            gender.append(0) # M
        else:
            gender.append(2) # OTHER
    return gender


def text_processing(messages, max_features = None):
    corpus = [re.sub(r"\b\d+\b|![:alpha:]|\d|\W|_|\b\w{1,2}\b", " ", m).lower() for m in messages]
    if max_features:
        vectorizer = CountVectorizer(strip_accents = 'ascii', stop_words=['english'], max_features = max_features)
    else:
        vectorizer = CountVectorizer(strip_accents = 'ascii', stop_words=['english'])
    vcorpus = vectorizer.fit_transform(corpus)
    labels = vectorizer.get_feature_names()
    return vcorpus, labels


def data_processing(data, vcorpus, column_class='', colnames=list):
    data[colnames] = data[colnames].astype(float)
    train = scale(data[colnames].values)
    target = data[column_class].values
    train = np.column_stack((vcorpus.toarray(), train))
    return train, target


def svm_cross_val(X, y, k=21, cv=5, C=0):
    svm_scores = {}
    for i in np.linspace(C, 1.0, k):
        clf = LinearSVC(C=i)
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        svm_scores[i] = scores.mean()
    return svm_scores


def plot_svm_scores(svm_score, savefig = True):
    x_values = np.array(list(svm_score.keys()))
    y_values = np.array(list(svm_score.values()))
    plt.figure(figsize=(12,12))
    plt.plot(x_values, y_values, marker="o", markerfacecolor="b")
    plt.plot(x_values[np.argmax(y_values)], y_values[np.argmax(y_values)], 'ro')
    plt.yticks(y_values)
    if savefig:
        plt.savefig('svm_scores.png' , dpi=300)
    else:
        plt.show()


def knn_cross_val(X, y, k=21, cv=5, step=1):
    knn_scores = {}
    for _k in range(1, k, step):
        knn = KNeighborsClassifier(n_neighbors = _k)
        scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
        knn_scores[_k] = scores.mean()
    return knn_scores


def plot_knn_scores(knn_scores, savefig = True):
    x_odd = np.array(list(knn_scores.keys()))[::2]
    y_odd = np.array(list(knn_scores.values()))[::2]
    x_even = np.array(list(knn_scores.keys()))[1::2]
    y_even = np.array(list(knn_scores.values()))[1::2]
    plt.figure(figsize=(12 , 12))
    plt.plot(x_odd , y_odd , marker="o" , markerfacecolor="b")
    plt.plot(x_even , y_even , marker="s" , markerfacecolor="r")
    plt.xlim(0 , 21)
    plt.xticks(range(0 , 21 , 1))
    plt.yticks(y_even)
    if savefig:
        plt.savefig('knn_scores.png' , dpi=300)
    else:
        plt.show()


"""
KNN IMPLEMENTATION
"""


class KnnClassifier:
    
    def __init__(self,n_neighbors):
        self.n_neighbors = n_neighbors
        self._X = None
        self._y = None
        self.pred = None

    def fit(self, X, y):
        if self.n_neighbors > X.shape[0]:
            raise ValueError('K neighbors greater than Training sample')
        self._X = X
        self._y = y

    def predict( self , X ):

        if X.shape[0] > self._X.shape[0]:
            raise ValueError('Test set (rows) greater than Train set')
        if X.shape[1] != self._X.shape[1]:
            raise ValueError('Test and Train set have different shape')

        self.pred = np.empty(shape=(X.shape[0], 0) , dtype=self._y.dtype)

        for i in range(X.shape[0]):
            _test = X[i,:]
            _euclidean = []
            _class = []
            for n in range(self._X.shape[0]):
                dist = np.sqrt(np.sum(np.square(_test - self._X[n,:])))
                _euclidean.append(dist)

            idx = np.argsort(_euclidean)

            for k in range(self.n_neighbors):
                _class.append(self._y[idx[k]])

            self.pred = np.append(self.pred, Counter(_class).most_common(1)[0][0])

        return self.pred

"""
MAIN - RUNS ALL THE FUNCTIONS GENERATES OUTPUT TO CONSOLE
"""

def main():

    """
    OPEN DATASET AND EXTRACT RELEVANT COLUMNS
    """

    t0 = time.time()

    data = pd.read_csv("humansofnewyork.csv")
    names = list(data.columns)
    colnames = [names[2] , names[5] , names[7] , names[8] , names[9]]
    data = data[data['message'].notnull()]

    """
    DATA PREPARATION AND PROCESSING
    """

    messages = data['message']
    gender = data_classes(messages)
    data['gender'] = gender
    drop = data[data['gender'] == 2]
    data = data.drop(drop.index)
    data = data.iloc[:, [2,5,7,8,9,10]]
    vcorpus, labels = text_processing(data['message'])
    train, train_target = data_processing(data, vcorpus, column_class='gender', colnames=colnames[2:])
    X_train , X_test , y_train , y_test = train_test_split(train, train_target,
                                                           test_size=0.3, train_size=0.7)

    print("")
    print("[  %0.3f ms  ] LOG: DATA PREPARATION STEP " % (time.time() - t0))

    """
    SVM FEATURE AND MODEL SELECTION
    """
    
    model = LinearSVC(C=1)
    model.fit(X_train, y_train)
    feature_selection = SelectFromModel(model, prefit=True)
    X_train = feature_selection.transform(X_train)
    X_test = feature_selection.transform(X_test)
    
    print("[  %0.3f ms  ] LOG: SVM FEATURE SELECTION AND MODEL FIT " % (time.time() - t0))

    clf = LinearSVC(C=1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    print("[  %0.3f ms   ] LOG: SVM CROSS VALIDATION " % (time.time() - t0))

    """
    CROSS-VALIDATION SVM
    """
    
    svc_scores = svm_cross_val(X_train, y_train, k=21, C=0.01)
    plot_svm_scores(svc_scores)

    """
    KNN MODEL SELECTION
    """

    print("[  %0.3f ms  ] WARNING: KNN FIT AND VALIDATION MAY TAKE A COUPLE OF MINUTES" % (time.time() - t0))

    # OWN IMPLEMENTATION
    knn = KnnClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    kpred = knn.predict(X_test)

    print("[  %0.3f ms  ] LOG: KNN MODEL FIT " % (time.time() - t0))

    # KNN SKLEARN IMPLEMENTATION TO PERFORM CROSS VALIDATION
    knn_scores = knn_cross_val(X_train, y_train, k=21, step=1)
    plot_knn_scores(knn_scores)

    print("[  %0.3f ms ] LOG: KNN CROSS VALIDATION " % (time.time() - t0))

    """
    RESULTS - F1 SCORES
    """
    
    # Weighted Average of the Precision and Recall
    svm_f1 = f1_score(y_pred=pred, y_true=y_test, average='weighted')
    knn_f1 = f1_score(y_pred=kpred, y_true=y_test, average='weighted')

    print("")
    print("----------------- Model Evaluation -----------------")
    print("f1 Score Metric - Scale (0.0 to 1.0) - (WORSE, BEST)")
    print("")
    print("--------------- SVM Metric Evaluation --------------")
    print("SVM f1 Score: %0.4f" % svm_f1)
    print("")
    print("--------------- KNN Metric Evaluation --------------")
    print("KNN f1 Score: %0.4f" % knn_f1 )
    print("")


if __name__ == "__main__":
    main()
