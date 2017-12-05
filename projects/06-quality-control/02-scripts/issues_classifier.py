# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 22:19:54 2017

@author: Justin.Stuck
"""

#import nltk
#nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split

porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


from nltk.corpus import stopwords
stop = stopwords.words('english')

import pandas as pd
import numpy as np
#df = pd.read_csv('PooledTextDataset.csv', encoding='latin1')
df = pd.read_csv('pooled_responses_v2.csv', encoding='latin1')
df.fillna('', inplace=True)

'''
df['binary_label'] = df['PooledOEScore'].apply(lambda x: 1 if x>0 else 0)
df['three_class_labels'] = df['PooledOEScore'].apply(lambda x: 0 if x==0 else (2 if x>10 else 1))
df['new_three'] = df['PooledOEScore'].apply(lambda x: 0 if x==0 else (2 if x>=3 else 1))
df['spaced_three'] = df['PooledOEScore'].apply(lambda x: 0 if x<=1 else (4 if x>4 else 2)) #[0,1] U (1,4] U (4, inf)
'''
def score(x):
    if x>2:
        return 3
    elif x>1:
        return 2
    elif x>0:
        return 1
    else:
        return 0
        
df['multiclass'] = df['PooledOEScore'].apply(score) #[0,1] U (1,4] U (4, inf)

    
'''
# binary label ~83/17 train test split
X_train = df.loc[:30000, 'PooledResponses'].values
y_train = df.loc[:30000, 'binary_label'].values
X_test = df.loc[30000:, 'PooledResponses'].values
y_test = df.loc[30000:, 'binary_label'].values


# three classes
X_train = df.loc[:30000, 'PooledResponses'].values
y_train = df.loc[:30000, 'three_class_labels'].values
X_test = df.loc[30000:, 'PooledResponses'].values
y_test = df.loc[30000:, 'three_class_labels'].values
'''

'''
# raw data
X = np.asarray(df.loc[:, 'PooledResponses'].values)
y = np.asarray(df.loc[:, 'spaced_three'].values)
'''
X = np.asarray(df.loc[:, 'PooledResponses'].values)
y = np.asarray(df.loc[:, 'multiclass'].values)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.25)


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix


from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC


parameters_svm = {'vect__ngram_range': [ (1,2), (1,3)],
               'tfidf__use_idf': (True, False),
               'svm_clf__C': ( .5, 1, 5),
               #'svm_clf__penalty': ( 'l1', 'l2'),
               'svm_clf__class_weight': (None, "balanced"),
               
 }
'''
#This is the absolute best I've seen
#Best parameter set: {'svm_clf__C': 1} 
CV Accuracy: 0.595
Test Accuracy: 0.855
[[4945  361   63   43]
 [ 304 2547  178   25]
 [  22  335  117    5]
 [  28   81   24   43]]
'''
'''
Best parameter set for full: {'svm_clf__C': 1, 'svm_clf__class_weight': 'balanced', 'tfidf__use_idf': False, 'vect__ngram_range': (1, 3)} 
CV Accuracy: 0.632
Test Accuracy: 0.844
[[5828  466   53   26]
 [ 378 3144  251   21]
 [  26  497  171    2]
 [  35   86   30   47]]
'''

linear_svm_clf = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1,3), lowercase=True)),
        ('tfidf', TfidfTransformer(use_idf=False)),
        ("svm_clf", LinearSVC(C=1,
                              loss='hinge', 
                              class_weight='balanced',
                              random_state=1))
    ])

#gets the best class by weighted f1 score
'''
gs_clf_svm = GridSearchCV(linear_svm_clf, parameters_svm, 
                          n_jobs=1,
                          scoring='f1_macro',
                          )
'''
clf_svm = linear_svm_clf.fit(X_train, y_train)

#persist it 
from sklearn.externals import joblib
joblib.dump(clf_svm, 'text_classifier.pkl')


'''
print('Best parameter set: %s ' % gs_clf_svm.best_params_)
print('CV Accuracy: %.3f' % gs_clf_svm.best_score_)

clf = gs_clf_svm.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

# Now for the confusion matrix
y_test_pred = cross_val_predict(clf, X_test, y_test, cv=5)
print (confusion_matrix(y_test, y_test_pred))
'''




'''
parameters_svm = {'vect__ngram_range': [(1,2)],
               #'tfidf__use_idf': (True, False),
               'clf-svm__alpha': ( 1e-3, 1,10),
               #'clf-svm__penalty': ( 'l1', 'l2'),
               #'clf-svm__class_weight': (None, "balanced"),
 }


text_clf_svm = Pipeline([('vect', CountVectorizer(lowercase=True)),
                      ('tfidf', TfidfTransformer()),
                      ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42,
                                            class_weight="balanced")),
 ])

gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=1, scoring='f1_macro')
gs_clf_svm = gs_clf_svm.fit(X_train, y_train)
print('Best parameter set: %s ' % gs_clf_svm.best_params_)
print('CV Accuracy: %.3f' % gs_clf_svm.best_score_)

clf = gs_clf_svm.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

# Now for the confusion matrix
y_test_pred = cross_val_predict(clf, X_test, y_test, cv=5)
print (confusion_matrix(y_test, y_test_pred))
'''
#
## Three class heuristic has ~91.7% accuracy
## using {'clf-svm__alpha': 0.0001, 'tfidf__use_idf': True, 'vect__ngram_range': (1, 1)} 
## 91.8 using {'clf-svm__alpha': 0.0001, 'clf-svm__penalty': 'l2', 'tfidf__use_idf': True, 'vect__ngram_range': (1, 2)}
#
## New three has ~92.0% accuracy 
## using {'clf-svm__alpha': 0.0001, 'tfidf__use_idf': True, 'vect__ngram_range': (1, 1)} 
## most satisfying confusion matrix yet with {'clf-svm__alpha': 0.0001, 'clf-svm__penalty': 'l2', 'tfidf__use_idf': True, 'vect__ngram_range': (1, 2)}
#'''
#

#'''
#pipeline.fit(X_train, y_train)
#y_predicted = pipeline.predict(X_test)
#from sklearn import metrics
#print(metrics.classification_report(y_test, y_predicted))
#
#clf = pipeline.steps[1][1]
#vect = pipeline.steps[0][1]
#for i, class_label in enumerate([0,1,2]):
#            topt = np.argsort(clf.coef_[i])[-15:]
#            print("%s:    %s" % (class_label,
#                  ", ".join(vect.get_feature_names()[j] for j in topt)))
#'''