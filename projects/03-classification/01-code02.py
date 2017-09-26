#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 18:18:06 2017

@author: jlroo
"""

import numpy as np

### Implementation of the Perceptron Algorithm 
### source: Python Machine Learning
### By Sebastian Raschka

class Perceptron(object):
    """Perceptron classifier.

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
    errors_ : list
        Number of misclassifications (updates) in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """Fit training data.

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
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    
############

"""
1 LINEAR SEPARABLE DATASET
"""

## Creates two normal populations of n=1000

np.random.seed(12345)
class_a_mean = -1.5
class_a_sigma = 1
class_b_mean = 2.5
class_b_sigma = 0.80

population_class_a = 1000
population_class_b = 1000

class_a = np.random.normal(class_a_mean, class_a_sigma, population_class_a)
class_b = np.random.normal(class_b_mean, class_b_sigma, population_class_b)


### Creates a dataset by randomly choosing
### from the two normal populations

sample_size = 100

feature_a_0 = np.random.choice(class_a, size = sample_size) / 0.35
feature_a_1 = np.random.choice(class_a, size = sample_size) + 0.20
label_a = np.repeat(1, sample_size)
sample_a = np.concatenate(([feature_a_0],[feature_a_1],[label_a]),axis=0)

feature_b_0 = np.random.choice(class_b, size = sample_size)
feature_b_1 = np.random.choice(class_b, size = sample_size)
label_b = np.repeat(-1, sample_size)
sample_b = np.concatenate(([feature_b_0],[feature_b_1],[label_b]),axis=0)

dataset = np.column_stack((sample_a,sample_b))

np.savetxt("data/data-separable.csv", dataset.T,fmt='%1.4f',
           delimiter=',' , comments='',
           header="feature_0,feature_1,class_label")


### Now with the linearly separable dataset created
### we can train the Perceptron

"""
PERCEPTRON TRAINING SEPARABLE DATASET
"""

X_train = dataset[:2].T
y_train = dataset[2]

ppn = Perceptron(eta=0.45, n_iter=10)
ppn.fit(X_train, y_train)

### Check if the Perceptron converges after 10
### iterations

print("\nPERCEPTRON TRAINING SEPARABLE DATASET")
print(" _____________________\
    \n| No. Errors | Epochs | \
    \n _____________________")
for i,e in enumerate(ppn.errors_):
    print( "|      " + str(e) + "     |    " + str(i) + "   |")
print("_____________________")


"""
2. NON-SEPARABLE DATASET
"""

### Same process as the first dataset
### Randomly sampling from the two normal populations

sample_size = 100

feature_a_0 = 5 + np.random.choice(class_a, size = sample_size) 
feature_a_1 = 2 + np.random.choice(class_a, size = sample_size) * 2
label_a = np.repeat(1, sample_size)
sample_a = np.concatenate(([feature_a_0],[feature_a_1],[label_a]),axis=0)

feature_b_0 = np.random.choice(class_b, size = sample_size) 
feature_b_1 = np.random.choice(class_b, size = sample_size) + 0.5
label_b = np.repeat(-1, sample_size)
sample_b = np.concatenate(([feature_b_0],[feature_b_1],[label_b]),axis=0)

dataset = np.column_stack((sample_a,sample_b))

np.savetxt("data/data-non-separable.csv", dataset.T,fmt='%1.4f',
           delimiter=',' , comments='',
           header="feature_0,feature_1,class_label")


"""
PERCEPTRON TRAINING NON-SEPARABLE DATASET
"""

X = dataset[:2].T
y = dataset[2]
ppn = Perceptron(eta=0.45, n_iter=10)
ppn.fit(X, y)

print("\nPERCEPTRON TRAINING NON-SEPARABLE DATASET")
print(" _____________________\
    \n| No. Errors | Epochs | \
    \n _____________________")
for i,e in enumerate(ppn.errors_):
    print( "|      " + str(e) + "     |    " + str(i) + "   |")
print("_____________________")



