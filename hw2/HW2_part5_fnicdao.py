"""
  Fiona Nicdao  COMP 479
  Homework 2: Percepton algorithm and Adaline algorithm

    PLEASE LOOK AT WORD DOC FOR REPORT on Saki 

    Part 5: Baseline Model
"""
import numpy as np # linear algebra
import pandas as pd # data processing / CVS file I/O
import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap
from Models import BaselineAdalineGD, TitanicDataset # my perceptron algorithm in Models.py 
    # and function for preprocessing Titanic Dataset

# import train.csv file 
## please change the file_path to where you store your train.csv file 
file_path = '/Users/fionanicdao/loyola/machineLearning/titanic_hw1/train.csv'
df = pd.read_csv(file_path)

# preprocess data : remove unused data and convert string values to numeric values
df = TitanicDataset.process_data(df)

#  shuffle the examples and split data : training 0.7 and test 0.3
df = df.sample(frac = 1)

#  shuffle the examples and split data : training 0.7 and test 0.3
df = df.sample(frac = 1)
ratio = 0.7 
total_rows = df.shape[0]
df_size = int(total_rows*ratio)
df_train = df[0:df_size]
df_test = df[df_size:]

"""
Training Dataset
"""
# set X(features) and y (target class labels) from dataset
y = df_train["Survived"].values
# X = df_train.iloc[0:,[1,5]].values
X = df_train.iloc[0:,1:5].values

# train training dataset with Adaline model 
ada_gd = BaselineAdalineGD(n_iter=300, eta=0.0001)
ada_gd.fit(X, y)
print("Training dataset's sum of squared error = " + str(ada_gd.cost_[-1]))

# plot Misclassification errors vs the number of epochs for training dataeset
plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.tight_layout()
plt.show()

"""
Test Dataset 
"""
# set X(features) and y (target class labels) from dataset
y_test = df_test["Survived"].values
X_test = df_test.iloc[0:,1:5].values

# train testing dataset with Adaline model
ada_gd_test = BaselineAdalineGD(n_iter=1000, eta=0.0001)
ada_gd_test.fit(X_test, y_test)
print("Test dataset's sum of squared error = " + str(ada_gd_test.cost_[-1]))
# plot Misclassification errors vs the number of epochs for training dataeset
plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.tight_layout()
plt.show()