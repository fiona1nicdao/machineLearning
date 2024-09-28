"""
  Fiona Nicdao  COMP 479
  Homework 2: Percepton algorithm and Adaline algorithm

    PLEASE LOOK AT WORD DOC FOR REPORT on Saki 

    PART 2: NON-Linearly Separate Dateset 
"""
import numpy as np  # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing / CVS file I/O
from matplotlib.colors import ListedColormap # for color map 
from Models import Perceptron, Plot # my perceptron algorithm in Models.py 
    # and function for decision region plot

#import nonlinear4_dataset.csv file 
## please change the file_path to where you store nonlinear_dataset.csv file
file_path = '/Users/fionanicdao/loyola/machineLearning/hw2/nonlinear_dataset.csv'
df_nonlinear = pd.read_csv(file_path)

# set X(features) and y (target class labels) from dataset 
y_nonlin = df_nonlinear.iloc[0:,2].values
y_nonlin = np.where( y_nonlin == "enoki ", -1,1)
X_nonlin = df_nonlinear.iloc[0:,[0,1]].values

# scatter plot data
plt.scatter(X_nonlin[:15,0], X_nonlin[:15,1], color='red',marker='o',label='portobello')
plt.scatter(X_nonlin[15:30,0],X_nonlin[15:30,1],color='blue',marker='x',label='enoki')
plt.xlabel('stem height[cm]')
plt.ylabel('cap width [cm]')
plt.legend(loc='upper right')
plt.show()

# train dataset with Perceptron model
ppn_nonlinear =Perceptron(learning_rate=0.1,epoch=20)
ppn_nonlinear.fit(X_nonlin,y_nonlin)
# calcuate accuracy 
accuracy = ppn_nonlinear.accuracy(y_nonlin)
print("\n\n accuracy of perceptron model with my nonlinear dataset is " + 
      str(accuracy) + "% \n\n" )

# plot Misclassification errors vs the number of epochs 
plt.plot(range(1, len(ppn_nonlinear.errors_)+1),ppn_nonlinear.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

# descision region plot - shows linearly separble 
Plot.plot_decision_regions(X_nonlin,y_nonlin, classifer=ppn_nonlinear)
plt.xlabel('cap length[cm]')
plt.ylabel('stem height [cm]')
plt.legend(loc='upper right')
plt.show()