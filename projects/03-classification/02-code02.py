#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 18:18:06 2017

@author: jlroo
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from numpy.random import seed
import numpy as np

### Implementation of the Perceptron Algorithm 
### source: Python Machine Learning
### By Sebastian Raschka

class AdalineSGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum-of-squares cost function value averaged over all
        training samples in each epoch.
    shuffle : bool (default: True)
        Shuffles training data every epoch if True to prevent cycles.
    random_state : int (default: None)
        Set random state for shuffling and initializing the weights.
        
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)
        
    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """Fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """Shuffle training data"""
        r = np.random.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Initialize weights to zeros"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True
        
    def _update_weights(self, xi, target):
        """Apply Adaline learning rule to update the weights"""
        output = self.net_input(xi)
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)


"""
3. TITANIC DATASET PREDICTION - USING AdelineSGD
"""

### Frist load and clean the data

train = open("../data/titanic.csv")
train = train.readlines()
train = [ item.replace('""',"") for item in train ]
train = [ item.split("\"") for item in train ]
train = [ item[0][:-1] + item[2] for item in train[1:] ]
train = [ item.split(",")[:7] + item.split(",")[8:] for item in train ]
train = [ b",".join(i.encode() for i in item) for item in train ]

train = np.genfromtxt(train, delimiter=",",
                     dtype=[('passengerid', '<i8'), ('survived', '<i8'), ('pclass', '<i8'),
                            ('sex', '|S1'), ('age', '<f8'), ('sibsp', '<i8'), ('parch', '<i8'),
                            ('fare', '<f8'), ('cabin', '|S1'), ('embarked', 'S1')],
                     names=['passengerid', 'survived', 'pclass', 'sex', 'age',
                            'sibsp', 'parch', 'fare', 'cabin', 'embarked'])

sex_classes, train['sex'] = np.unique(train['sex'], return_inverse=True)
cabin_classes, train['cabin'] = np.unique(train['cabin'], return_inverse=True)
embarked_classes, train['embarked'] = np.unique(train['embarked'], return_inverse=True)

train = train.astype([  ('passengerid', '<i8'), ('survived', '<i8'), ('pclass', '<i8'),
                        ('sex', '<i8'), ('age', '<f8'), ('sibsp', '<i8'), ('parch', '<i8'),
                        ('fare', '<f8'), ('cabin', '<i8'), ('embarked', '<i8')])

colnames = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'cabin', 'embarked']

for feature in colnames:
    train[feature] = np.where(np.isnan(train[feature]), np.nanmean(train[feature], axis=0), train[feature])

"""
0.1 FEATURE SELECTION - ALL FEATURES
"""

### From all the features in the data we select the onces with 
### most predictive power starting with all features

pred = np.array([train['pclass'], train['sex'], train['age'], train['sibsp'], train['parch'], train['fare'], train['cabin'], train['embarked']])
pred = pred.transpose()
pred0 = normalize(pred)

passengerId = train['passengerid']
train_target = train['survived']

X_train0, X_test0,\
y_train0, y_test0,\
idx_train0, idx_test0 = train_test_split(pred0, train_target, 
                                       passengerId, test_size = 0.3, 
                                       train_size= 0.7)


### All models will have the same iterations
### and error rate to measure performance

epochs = 100
error_rate = 0.0001

"""
0.2 TRAINING THE MODEL ALGORITHM - ALL FEATURES
"""
ada0 = AdalineSGD(n_iter = 100, eta = 0.0001)
ada0.fit(X_train0, y_train0)
pred0 = ada0.predict(X_test0)
mse0 = mean_squared_error(y_pred = pred0, y_true = y_test0)
cost0 = ada0.cost_[-1]

"""
1.1 FEATURE SELECTION - AGE, PCLASS, SEX, FARE
"""

pred = np.array([train['age'], train['pclass'], train['sex'], train['fare']])
pred = pred.transpose()
pred1 = normalize(pred)
passengerId = train['passengerid']
train_target = train['survived']

X_train1, X_test1,\
y_train1, y_test1,\
idx_train1, idx_test1 = train_test_split(pred1, train_target, 
                                       passengerId, test_size = 0.3, 
                                       train_size= 0.7)

"""
1.2 TRAINING THE MODEL ALGORITHM - AGE, PCLASS, SEX, FARE
"""
ada1 = AdalineSGD(n_iter = 100, eta = 0.0001)
ada1.fit(X_train1, y_train1)
pred1 = ada1.predict(X_test1)
mse1 = mean_squared_error(y_pred = pred1, y_true = y_test1)
cost1 = ada1.cost_[-1]

"""
2.1 FEATURE SELECTION - AGE, PCLASS
"""

pred = np.array([train['age'], train['pclass']])
pred = pred.transpose()
pred2 = normalize(pred)
passengerId = train['passengerid']
train_target = train['survived']
X_train2, X_test2,\
y_train2, y_test2,\
idx_train2, idx_test2 = train_test_split(pred2, train_target, 
                                       passengerId, test_size = 0.3, 
                                       train_size= 0.7)


"""
2.2 TRAINING THE MODEL ALGORITHM - AGE, PCLASS
"""

ada2 = AdalineSGD(n_iter = 100, eta = 0.0001)
ada2.fit(X_train2, y_train2)
pred2 = ada2.predict(X_test2)
mse2 = mean_squared_error(y_pred = pred2, y_true = y_test2)
cost2 = ada2.cost_[-1]

"""
3.1 FEATURE SELECTION - PCLASS, SEX
"""

pred = np.array([train['pclass'], train['sex']])
pred = pred.transpose()
pred3 = normalize(pred)
passengerId = train['passengerid']
train_target = train['survived']
X_train3, X_test3,\
y_train3, y_test3,\
idx_train3, idx_test3 = train_test_split(pred3, train_target, 
                                       passengerId, test_size = 0.3, 
                                       train_size= 0.7)


"""
3.2 TRAINING THE MODEL ALGORITHM - PCLASS, SEX
"""

ada3 = AdalineSGD(n_iter = 100, eta = 0.0001)
ada3.fit(X_train3, y_train3)
pred3 = ada3.predict(X_test3)
mse3 = mean_squared_error(y_pred = pred3, y_true = y_test3)
cost3 = ada3.cost_[-1]

"""
4.1 FEATURE SELECTION - AGE, FARE
"""

pred = np.array([train['age'], train['fare']])
pred = pred.transpose()
pred4 = normalize(pred)
passengerId = train['passengerid']
train_target = train['survived']

X_train4, X_test4,\
y_train4, y_test4,\
idx_train4, idx_test4 = train_test_split(pred4, train_target, 
                                       passengerId, test_size = 0.3, 
                                       train_size= 0.7)


"""
4.2 TRAINING THE MODEL ALGORITHM - ALL FEATURES
"""

ada4 = AdalineSGD(n_iter = 100, eta = 0.0001)
ada4.fit(X_train4, y_train4)
pred4 = ada4.predict(X_test4)
mse4 = mean_squared_error(y_pred = pred4, y_true = y_test4)
cost4 = ada4.cost_[-1]


print("\nPERFORMANCE ON TITANIC DATASET \n")
print("  Learning Rate : " + str(error_rate))
print("  No. Iterations: " + str(epochs))
print("                                ")
print("|AdelineSGD Model |    Average Cost  |        MSE        |")
print("------------------|------------------|-------------------|")
print("|      ada0       |       %6.4f"%(cost0) + "     |       %6.4f"%(mse0)+"      |")
print("|      ada1       |       %6.4f"%(cost1) + "     |       %6.4f"%(mse1)+"      |")
print("|      ada2       |       %6.4f"%(cost2) + "     |       %6.4f"%(mse2)+"      |")
print("|      ada3       |       %6.4f"%(cost3) + "     |       %6.4f"%(mse3)+"      |")
print("|      ada4       |       %6.4f"%(cost4) + "     |       %6.4f"%(mse4)+"      |")
 
