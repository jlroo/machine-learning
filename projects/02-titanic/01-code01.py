#! env/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 1:00:00 2017
Jose Luis Rodriguez
@author: jlroo
"""

import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split

###################### TRAINING DATA ####################

train = open("data/train.csv")
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

predictors = np.array([train['pclass'], train['sex'], train['age'], train['sibsp'],
                       train['parch'], train['fare'], train['cabin'], train['embarked']])

passengerId = train['passengerid']
predictors = predictors.transpose()
train_target = train['survived']

##################### LINEAR MODEL ####################

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'cabin', 'embarked']

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(predictors, train_target, passengerId)
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)

predict_train = model.predict(X_test)

print("Model Equation: \n")
coef = model.coef_[0]
intercept = model.intercept_[0]

equation = "".join("("+str(coef[i])+"*"+k+")+" for i,k in enumerate(features))
equation = "Survived = " + str(intercept)+" + " + equation
print(equation + "\n")

score = model.score(X_test, y_test)
print("Model Score: " + str(score) +"\n")

##################### PREDICTION OUT ####################

passengerId = np.asarray(list(idx_test))
predict_train = np.asarray(list(predict_train))
target = np.asarray(list(y_test))

training_model = np.array([passengerId, predict_train, target], dtype='<i8')
training_model = training_model.transpose()

np.savetxt("predicted_train.csv", training_model, fmt='%i', delimiter=",",
           header="passengerId,predicted_survival,target_survival")

###################### TEST DATA SET #################

test = open("data/test.csv")
test = test.readlines()
test = [ item.replace('""', "") for item in test ]
test = [ item.split("\"") for item in test ]
test = [ item[0][:-1] + item[2] for item in test[1:] ]
test = [ item.split(",")[:6] + item.split(",")[7:] for item in test ]
test = [ b",".join(i.encode() for i in item) for item in test ]

test = np.genfromtxt(test,
                     delimiter=",",
                     dtype=[('passengerid', '<i8'), ('pclass', '<i8'), ('sex', '|S1'),
                            ('age', '<f8'), ('sibsp', '<i8'), ('parch', '<i8'),
                            ('fare', '<f8'), ('cabin', '|S1'), ('embarked', 'S1')],
                     names=['passengerid', 'pclass', 'sex', 'age', 'sibsp',
                            'parch', 'fare', 'cabin', 'embarked'])


sex_classes, test['sex'] = np.unique(test['sex'], return_inverse=True)
cabin_classes, test['cabin'] = np.unique(test['cabin'], return_inverse=True)
embarked_classes, test['embarked'] = np.unique(test['embarked'], return_inverse=True)

test = test.astype([('passengerid', '<i8'), ('pclass', '<i8'), ('sex', '<i8'),
                    ('age', '<f8'), ('sibsp', '<i8'), ('parch', '<i8'),
                    ('fare', '<f8'), ('cabin', '<i8'), ('embarked', '<i8')])

for feature in colnames:
    test[feature] = np.where(np.isnan(test[feature]), np.nanmean(test[feature], axis=0), test[feature])

test_predictors = np.array([test['pclass'], test['sex'], test['age'], test['sibsp'],
                            test['parch'], test['fare'], test['cabin'],  test['embarked']])

test_predictors = test_predictors.transpose()

### PREDICTION ON TESTING DATA
predict_test = model.predict(test_predictors)

predict_test = np.asarray(list(predict_test))
testing_model = np.array([test["passengerid"], predict_test], dtype='<i8')
testing_model = testing_model.transpose()

np.savetxt("predicted_Submission.csv", testing_model, fmt='%i', delimiter=",", header="passengerId,predicted_survival")
