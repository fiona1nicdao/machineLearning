"""
    Baseline Approach Description 
    Goal: to establish the performance of a baseline system apply the Dummy-Classifer
from scikit-learn (sklearn.dummy.DummyClassifier) to your data. At minimum, try the 
following two values for the 'strategy' 
parameter:"most_frequent", "prior", "stratified", "uniform"
"""
from sklearn.dummy import DummyClassifier
import numpy as np 
import pandas as pd  # data processing / CVS file I/O
import csv
import matplotlib.pyplot as plt

file_path_xtrain = '/Users/fionanicdao/loyola/machineLearning/project/X_train.csv'
X_train = pd.read_csv(file_path_xtrain)

file_path_ytrain = '/Users/fionanicdao/loyola/machineLearning/project/y_train.csv'
y_train = pd.read_csv(file_path_ytrain)

scores =[]
strategy = ["most_frequent", "prior", "stratified", "uniform"]
for s in strategy :
    dummy_clf = DummyClassifier(strategy=s)
    dummy_clf.fit(X_train, y_train)
    score = dummy_clf.score(X_train, y_train)
    scores.append(score)
# Visualize accuracy vs. dummy classifier
fig, ax = plt.subplots()
ax.plot(strategy, scores)
ax.set(xlabel="strategy",
       ylabel="Accuracy",
       title="Performance of Dummy Classifier")
plt.show()