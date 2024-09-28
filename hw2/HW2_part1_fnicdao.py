"""
  Fiona Nicdao  COMP 479
  Homework 2: Percepton algorithm and Adaline algorithm
    Titanic dataset from Kaggle.com : https://www.kaggle.com/c/titanic

    PLEASE LOOK AT WORD DOC FOR REPORT on Saki 

    PART 1: Linearly Separate Dateset 
"""

import numpy as np  # linear algebra
import pandas as pd # data processing / CVS file I/O
import matplotlib.pyplot as plt # for scatter plot 
from matplotlib.colors import ListedColormap # for color map 
from Models import Perceptron, Plot # my perceptron algorithm in Models.py 
    # and function for decision region plot

#import linear2_dataset.csv file 
## please change the file_path to where you store linear_dataset.csv file
file_path = '/Users/fionanicdao/loyola/machineLearning/hw2/linear_dataset.csv'
df = pd.read_csv(file_path)

# set X(features) and y (target class labels) from dataset 
y = df.iloc[0:,2].values
y = np.where(y == "enoki ", -1,1)
X = df.iloc[0:,[0,1]].values


plt.scatter(X[:15,1], X[:15,0], color='red',marker='o',label='portobello')
plt.scatter(X[15:30,1],X[15:30,0],color='blue',marker='x',label='enoki')
plt.xlabel('gills present / absent')
plt.ylabel('cap length[cm]')
plt.legend(loc='upper left')
plt.show()

# train dataset with Perceptron model 
ppn = Perceptron(learning_rate=0.1,epoch=20)
ppn.fit(X,y)
# calcuate accuracy 
accuracy = ppn.accuracy(y)
print("\n\n accuracy of perceptron model with my linear dataset is " + 
      str(accuracy) + "% \n\n" )

# plot Misclassification errors vs the number of epochs 
plt.plot(range(1, len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

# descision region plot - shows linearly separble 
Plot.plot_decision_regions(X,y, classifer=ppn)
plt.xlabel('cap length [cm]')
plt.ylabel('gills present / absent')
plt.legend(loc='upper left')
plt.show()