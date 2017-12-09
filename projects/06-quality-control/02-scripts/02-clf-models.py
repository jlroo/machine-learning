#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    COMP 488 - Machine Learning
    Applied Machine Learning to Quality Control Workflow
    Jose Luis Rodriguez
    Created on Fri Dec 09 18:00:00 2017
    @author: jlroo
    
    KNN, Random Forest and LinearSVC
    Models
"""


import time
import itertools
import numpy as np
import pandas as pd
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt

from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import train_test_split, cross_val_predict

from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix


# In[2]:

def summary(clf, X_test, y_test):
    print('        Accuracy Score: %.3f' % clf.score(X_test, y_test))
    print('      Model Best Score: %.3f' % clf.best_score_)
    print('   Best Parameters Set: %s\n' % clf.best_params_)
    print('-------- Best Estimator -------- \n%s' % clf.best_estimator_)


def model_metrics(clf, X_test, y_test, cv=5, grid_search=False):
    if grid_search:
        y_pred = clf.predict(X_test)
        best_score_ = clf.best_score_
    else:
        y_pred = cross_val_predict(clf, X_test, y_test, cv=cv)
        best_score_ = cross_val_score(clf, X_test, y_test, cv=cv).mean()
    cmatrix = confusion_matrix(y_test, y_pred)
    score = f1_score(y_test, y_pred, average='macro')
    metrics = {'score': score,
               'cv_score':best_score_,
               'confusion_matrix': cmatrix,
               'y_pred': y_pred}
    return metrics
    
def model_metrics(clf, X_test, y_test,
                  cv = 5,
                  target_names = None,
                  grid_search = True):
    report = None
    if grid_search:
        y_pred = clf.predict(X_test)
        best_score_ = clf.best_score_
    else:
        y_pred = cross_val_predict(clf, X_test, y_test, cv=cv)
        best_score_ = cross_val_score(clf, X_test, y_test, cv=cv).mean()
    if target_names:
        report = classification_report(y_test, y_pred, target_names=target_names)
        
    cmatrix = confusion_matrix(y_test, y_pred)
    roc_score = roc_auc_score(y_test, y_pred)
    pre_score = precision_score(y_test, y_pred)
    rec_score = recall_score(y_test, y_pred)
    score = f1_score(y_test, y_pred, average='macro')
    metrics = {
        'score': score,
        'roc_score': roc_score,
        'pre_score': pre_score,
        'rec_score': rec_score,
        'confusion_matrix': cmatrix,
        'report': report,
        'y_pred': y_pred
    }
    return metrics


def plot_roc(clf, X_test, y_test):
    y_score = clf.predict_proba(X_test)
    n_classes = y_test.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    

def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix',
                          cmap = plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def data_split(X,y,splits = [0.3,0.5]):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = splits[0], 
                                                        random_state = 1)
    X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, 
                                                    test_size = splits[1], 
                                                    random_state = 1)
    return X_train, y_train, X_dev, y_dev, X_test, y_test


# ### Data Processing

# In[3]:


responses = pd.read_csv("responses.csv",sep=",")


# In[4]:


responses.head()


# In[5]:


X = responses[['top_3pct_time', 'speeding',
               'navigation_rating', 'frame_score',
               'straight_line', 'no_pick_ups',
               'outlier_sales', 'not_in_database',
               'num_open_ends', 'class_issue']]

y = responses['disqualified'].as_matrix()


# In[6]:


X_train, y_train, X_dev, y_dev, X_test, y_test = data_split(X, y, splits = [0.3,0.5])


# In[7]:


scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_dev = scaler.transform(X_dev)


# ### Model KNeighbors

# In[8]:


n_start = 2
n_end = 7
p_start = 1
p_end = 3

params_knn = {'n_neighbors': np.arange(n_start,n_end,1),
              'p': np.arange(p_start,p_end,1)}


# In[9]:


knn_clf = KNeighborsClassifier()


# In[10]:

gs_clf = GridSearchCV(knn_clf,  params_knn,  n_jobs=-1,  scoring='f1_macro')
gs_clf.fit(X_train, y_train)


# In[11]:


summary(gs_clf, X_dev, y_dev)


# In[12]:


metrics_knn = model_metrics(gs_clf, X_test, y_test, cv=5,
                            target_names=["True Disqualified","False Disqualified"])


# In[13]:


print('       F1 Score: {}'.format(metrics_knn['score']))
print('   Recall Score: {}'.format(metrics_knn['rec_score']))
print('Precision Score: {}'.format(metrics_knn['pre_score']))
print('  ROC AUC Score: {}'.format(metrics_knn['roc_score']))


# In[14]:


print(metrics_knn['report'])


# In[15]:


y_pred = gs_clf.predict(X_test)


# In[16]:


# Compute confusion matrix
class_names = ['Qualified','Disqualified']
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)


# In[17]:


# Plot normalized confusion matrix
plt.figure(figsize=(8,8))
plot_confusion_matrix(cnf_matrix, 
                      classes = class_names, 
                      normalize = True,
                      title = 'KNeighbors - Confusion Matrix (Normalized)')

plt.show()

# Plot non-normalized confusion matrix
plt.figure(figsize=(8,8))
plot_confusion_matrix(cnf_matrix, 
                      classes=class_names, 
                      title='KNeighbors - Confusion Matrix')
plt.show()


# ### ROC Analysis - Stratified KFold Classification - KNeighbors

# In[18]:


knn_clf = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=5, p=2,
           weights='uniform')


# In[19]:


cv = StratifiedKFold(n_splits=6)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
#X_ = X.as_matrix()
X_ = X.as_matrix()

plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X_, y):
    probas_ = knn_clf.fit(X_[train], y[train]).predict_proba(X_[test])
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
    
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Stratified KFold ROC Plot - KNeighbors Classifier')
plt.legend(loc="lower right")
plt.show()


# ### Model Random Forest

# In[20]:


n_start = 5
n_end = 21
params_rf = {'n_estimators': np.arange(n_start, n_end), 
             'class_weight':(None, 'balanced'),
             'max_features':(None, 'auto', 'log2'),
             'criterion':('gini', 'entropy')}

rf_clf = RandomForestClassifier()


# In[21]:

gs_rf = GridSearchCV(rf_clf,  params_rf,  n_jobs=-1,  scoring='f1_macro')
gs_rf.fit(X_train, y_train)


# In[22]:


summary(gs_rf, X_dev, y_dev)


# In[23]:


metrics_rf = model_metrics(gs_rf, X_test, y_test, cv=5,
                            target_names=["True Disqualified",
                                          "False Disqualified"])

# In[24]:


print('       F1 Score: {}'.format(metrics_rf['score']))
print('   Recall Score: {}'.format(metrics_rf['rec_score']))
print('Precision Score: {}'.format(metrics_rf['pre_score']))
print('  ROC AUC Score: {}'.format(metrics_rf['roc_score']))


# In[25]:


print(metrics_rf['report'])


# ### Random Forest Confusion Matrix

# In[26]:


y_pred = gs_rf.predict(X_test)


# In[27]:


# Compute confusion matrix
class_names = ['Qualified', 'Disqualified']
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision = 2)


# In[28]:


# Plot normalized confusion matrix
plt.figure(figsize=(8,8))
plot_confusion_matrix(cnf_matrix, 
                      classes = class_names, 
                      normalize = True,
                      title = 'Random Forest - Confusion Matrix (Normalized)')

plt.show()

# Plot non-normalized confusion matrix
plt.figure(figsize=(8,8))
plot_confusion_matrix(cnf_matrix, 
                      classes=class_names, 
                      title='Random Forest - Confusion Matrix')
plt.show()


# ### ROC Analysis - Stratified KFold Classification - Random Forest

# In[29]:


rf_clf = RandomForestClassifier(bootstrap=True, class_weight=None,
                                criterion='entropy', max_depth=None,
                                max_features='log2', max_leaf_nodes=None,
                                min_impurity_decrease=0.0, min_impurity_split=None,
                                min_samples_leaf=1, min_samples_split=2,
                                min_weight_fraction_leaf=0.0, n_estimators=13,
                                n_jobs=1, oob_score=False,
                                random_state=None, verbose=0,
                                warm_start=False)

# In[30]:


cv = StratifiedKFold(n_splits=6)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
X_ = X.as_matrix()
plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X_, y):
    probas_ = rf_clf.fit(X_[train], y[train]).predict_proba(X_[test])
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
    
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Stratified KFold ROC Plot - Random Forest Classifier')
plt.legend(loc="lower right")
plt.show()


# ### Model LinearSVC

# In[31]:


params_svm = parameters_svm = {'C': ( .01, .1, 1, 10, 20),
               'loss':('hinge', 'squared_hinge'),
               'class_weight': (None, "balanced")}


# In[32]:


svm_clf = LinearSVC()


# In[33]:


gs_svm = GridSearchCV(svm_clf,  params_svm,  n_jobs=-1,  scoring='f1_macro')
gs_svm.fit(X_train, y_train)


# In[34]:


summary(gs_svm, X_dev, y_dev)


# In[35]:


metrics_svm = model_metrics(gs_svm, 
                            X_test, y_test, cv=5,
                            target_names=["True Disqualified","False Disqualified"])


# In[36]:


print('       F1 Score: {}'.format(metrics_svm['score']))
print('   Recall Score: {}'.format(metrics_svm['rec_score']))
print('Precision Score: {}'.format(metrics_svm['pre_score']))
print('  ROC AUC Score: {}'.format(metrics_svm['roc_score']))


# In[37]:


print(metrics_svm['report'])


# ### LinearSVC Confusion Matrix

# In[38]:


y_pred = gs_svm.predict(X_test)


# In[39]:


# Compute confusion matrix
class_names = ['Qualified','Disqualified']
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)


# In[40]:


# Plot normalized confusion matrix
plt.figure(figsize=(8,8))
plot_confusion_matrix(cnf_matrix, 
                      classes = class_names, 
                      normalize = True,
                      title = 'LinearSVC - Confusion Matrix (Normalized)')

plt.show()

# Plot non-normalized confusion matrix
plt.figure(figsize=(8,8))
plot_confusion_matrix(cnf_matrix, 
                      classes=class_names, 
                      title='LinearSVC - Confusion Matrix')
plt.show()


# ### ROC Analysis - Stratified KFold Classification -  LinearSVC

# In[41]:


svm_clf = LinearSVC(C=10, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='hinge', max_iter=1000, multi_class='ovr',
     penalty='l2', random_state=None, tol=0.0001, verbose=0)


# In[42]:


cv = StratifiedKFold(n_splits=6)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
X_ = X.as_matrix()
plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X_, y):
    probas_ = svm_clf.fit(X_[train], y[train]).decision_function(X_[test])
    fpr, tpr, thresholds = roc_curve(y[test], probas_)
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    i += 1
    
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Luck', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Stratified KFold ROC Plot - LinearSVC Classifier')
plt.legend(loc="lower right")
plt.show()

