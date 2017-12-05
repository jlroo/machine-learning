# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 16:34:35 2017

@author: Justin.Stuck
"""

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_predict
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC

from sklearn.externals import joblib

# load our full data set
data = pd.read_csv('full_pooled_responses_v2.csv', encoding='latin1', index_col='uuid')

# deserialize our text classifier
txt_clf = joblib.load('text_classifier.pkl')

# Use the text classifier to score the responses
data['OEScore'] = txt_clf.predict(np.asarray(data['PooledResponses'].values))

predictors = ['Top 3% Time', 'Speeding', 'Navigation Rating', 'Frame Score',
       'Straight-line', 'No Pick Ups', 'Outlier Sales', 'Not in Database',
        'NumOpenEnds', 'OEScore']

# Split the data into a train and test set
X = data[predictors].values
y = data.DQ.values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.2,
                                                    random_state=2)

# Preprocess
scaler = RobustScaler()
#scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#for timing
import time
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print ('Finished training in %2.2f seconds \n' % ((te - ts)))
        return result
    return timed


# need to look back through this one
import matplotlib.pyplot as plt
# Need to look back through this plotting definition
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=None)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0,1,0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    
def plot_precision_recall_curve(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0,1])
    plt.show()
    

    
def summary( classifier, X_test, y_test):
    print('Best parameter set: %s ' % classifier.best_params_)
    print('CV F1-Score: %.3f' % classifier.best_score_)
    
    clf = classifier.best_estimator_
    print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
    
    # Now for the confusion matrix
    y_test_pred = cross_val_predict(clf, X_test, y_test, cv=3)#, method="decision_function")
    #prec, rec, thresh = precision_recall_curve(y_test, y_test_pred)
    #plot_precision_recall_curve(prec, rec, thresh)
    
    print (confusion_matrix(y_test, y_test_pred))
    print ('ROC_AUC score: {}'.format(roc_auc_score(y_test, y_test_pred)))
    print('Precision: {} \nRecall: {} \nMacro F1-Score: {}'.format(
            precision_score(y_test, y_test_pred),
            recall_score(y_test, y_test_pred),
            f1_score(y_test, y_test_pred, average='macro')))
    
    #Now plot the ROC curve
    #fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
    #plot_roc_curve(fpr, tpr)

    return y_test_pred



@timeit
def test_clf(name, train_X, train_y, test_X, test_y, clf, params):
    print("Now training the {} model...".format(name))
    gs_clf = GridSearchCV(clf, params, 
                          n_jobs=1,
                          scoring='f1_macro', #this weights the true and false classes equally which helps to account for the class imbalance
                          )
    gs_clf.fit(train_X, train_y)
    return summary( gs_clf, test_X, test_y)



'''
# Define logistic regression parameters
params_lr = {'penalty':('l1','l2'),
             'C':(.1, 1, 10),
             'class_weight':(None, 'balanced'),
             }

lr_clf = LogisticRegression()

# Here we use roc auc in order to find the best fit
# fit the training data with the logistic regression model

test_clf("Logistic Regression", X_train, y_train, X_test, y_test, lr_clf, params_lr)
#Best parameter set: {'C': 1, 'class_weight': None, 'penalty': 'l1'} 
#CV Accuracy: 0.877
#Test Accuracy: 0.965
#[[7993   49]
# [ 268  537]]
#ROC_AUC score: 0.8304938668264902
#Precision: 0.9163822525597269 
#Recall: 0.6670807453416149 
#Macro F1-Score: 0.8763310621073168







# Now for the Random Forest classifier
from sklearn.ensemble import RandomForestClassifier

params_rf = {'n_estimators': (range(5, 20)), 
             'class_weight':(None, 'balanced'),
             'max_features':(None, 'auto', 'log2'),
             'criterion':('gini', 'entropy')}

rf_clf = RandomForestClassifier()

test_clf("Random Forest", X_train, y_train, X_test, y_test, rf_clf, params_rf)
# The roc is better for this one, with a roc_auc score of 86.03% as opposed to 82.9% for the logistic regression



from sklearn.ensemble import AdaBoostClassifier

rf_clf = RandomForestClassifier(class_weight=None,
                                 criterion='gini', 
                                 max_features='log2', 
                                 n_estimators=12)
ada_clf = AdaBoostClassifier(base_estimator=rf_clf, n_estimators=500,
                             learning_rate=.01,
                             random_state=0)
params_ada = {}
test_clf("Adaboosted Random Forest", X_train, y_train, X_test, y_test, ada_clf, params_ada)
# This was inefficient and didn't gain much for the complexity
#Now training the Adaboosted Random Forest model...
#Best parameter set: {} 
#CV Accuracy: 0.895
#Test Accuracy: 0.967
#[[7956   86]
# [ 243  562]]
#ROC_AUC score: 0.8437213943566463
#Precision: 0.8672839506172839 
#Recall: 0.698136645962733 
#Macro F1-Score: 0.8766572734253621
#Finished training in 560.80 seconds 



# Now for the LinearSVM classifier

params_svm = parameters_svm = {'C': ( .01, .1, 1, 10, 20),
               'loss':('hinge', 'squared_hinge'),
               'class_weight': (None, "balanced"),
 }

svm_clf = LinearSVC()
test_clf("Linear SVM", X_train, y_train, X_test, y_test, svm_clf, params_svm)
#Best parameter set: {'C': 1, 'class_weight': None, 'loss': 'squared_hinge'} 
#CV Accuracy: 0.971
#Test Accuracy: 0.963
#[[7994   48]
# [ 284  521]]
#roc_auc score: 0.820618152216392
#Test Precision: 0.9156414762741653 
#Test Recall: 0.6472049689440994
#
#Note that by slightly lowering the penalty here, we can reduce the false negatives
#by a factor of 4. This will lead to a more than 10-fold increase in false positives however
# As mentioned above, here is the run with those parameters
#Best parameter set: {'C': 0.1, 'class_weight': 'balanced', 'loss': 'squared_hinge'} 
#CV Accuracy: 0.971
#Test Accuracy: 0.922
#[[7457  585]
# [  75  730]]
#roc_auc score: 0.9170445997024936
#Test Precision: 0.5551330798479087 
#Test Recall: 0.906832298136646






#Now for the non-linear SVM classifier

params_svm = parameters_svm = {'C': ( 1, 10, 50, 100),
                               #'degree': (1, 2, 3) best was still the default of 3
                               'kernel':( 'poly', 'sigmoid', 'rbf'),
                               'class_weight': (None, "balanced"),
}

svm_clf = SVC(kernel='poly')#, class_weight='balanced')
test_clf("Non-linear SVM",X_train, y_train, X_test, y_test, svm_clf, params_svm)

#Best parameter set: {'C': 1, 'class_weight': 'balanced', 'kernel': 'poly'} 
#CV Accuracy: 0.971
#Test Accuracy: 0.948
#[[7691  351]
# [ 109  696]]
#roc_auc score: 0.9104752070264651
#Test Precision: 0.664756446991404
#Test Recall: 0.8645962732919255
#Best parameter set: {'C': 10, 'class_weight': None, 'kernel': 'poly'} 
#CV Accuracy: 0.895
#Test Accuracy: 0.969
#[[7959   83]
# [ 230  575]]
#ROC_AUC score: 0.8519824492841156
#Precision: 0.8738601823708206 
#Recall: 0.7142857142857143 
#Macro F1-Score: 0.8833859816028441





#We also need to do linear/quadratic discriminant analysis

# For LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA


params_lda = parameters_svm = {'solver': ( 'lsqr', 'eigen'),
                               'shrinkage':( 'auto', 0, .5, 1),
                               'n_components': (3, 5, 7, 10),
 }

lda_clf = LDA()
test_clf("LDA", X_train, y_train, X_test, y_test, lda_clf, params_lda)
# LDA is AWFUL
#Best parameter set: {'n_components': 3, 'shrinkage': 0.5, 'solver': 'lsqr'} 
#CV Accuracy: 0.959
#Test Accuracy: 0.935
#[[8042    0]
# [ 575  230]]
#roc_auc score: 0.6428571428571428



# Now for QDA
params_qda =  {}

qda_clf = QDA()
test_clf("QDA", X_train, y_train, X_test, y_test, qda_clf, params_qda)
# This wasn't great CV Accuracy: 0.938
#Test Accuracy: 0.931
#[[4539 3503]
# [ 249  556]]
#roc_auc score: 0.6275475338324726

'''

# and if we have time, let's toss a neural network at the problem
from sklearn.neural_network import MLPClassifier

params_mlp = {'solver': ('lbfgs', 'adam'),
              #'alpha':(1e-5),
              'hidden_layer_sizes':((10,), (10, 2), (10, 5, 2))
              }


mlp_clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(10, 5, 2), random_state=1)
test_clf("MLP Neural Network", X_train, y_train, X_test, y_test, mlp_clf, params_mlp)
# 
## with lbfgs and (10,2) hidden layers, CV Accuracy: 0.974
#Test Accuracy: 0.967
#[[7942  100]
# [ 195  610]]
#roc_auc score: 0.8726646287116859

# with (10, 5, 2) layers , i.e. 10 input input layer, 5 node hidden layer, and a 2 node output layer,

#CV Accuracy: 0.974
#Test Accuracy: 0.967
#[[7950   92]
# [ 192  613]]
#roc_auc score: 0.8750253714582293
# This one slightly outperformed some of the other techniques




# maybe a knn for kicks?
from sklearn.neighbors import KNeighborsClassifier

params_knn = {'n_neighbors': (4, 5, 6),
              'p':(1, 2),
              }


knn_clf = KNeighborsClassifier(p=2)
y_test_pred = test_clf("KNN", X_train, y_train, X_test, y_test, knn_clf, params_knn)
#Best parameter set: {'n_neighbors': 5, 'p': 2} 
#CV Accuracy: 0.891
#Test Accuracy: 0.967
#[[7938  104]
# [ 234  571]]
#ROC_AUC score: 0.8481923318725757
#Precision: 0.845925925925926 
#Recall: 0.7093167701863354 
#Macro F1-Score: 0.8753877196550195
#Finished training in 218.55 seconds 








