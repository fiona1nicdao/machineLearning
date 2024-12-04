""" 
    build a machine learning model 
    Model : neural network 
    Model performance evaluation choosen is ? 
        correct evaluation metric 
        test your model against a held out test set 
        the test set should be allocated before any ML experiements are condects with your data 
        
    
"""
import tensorflow as tf
from tensorflow import keras 
import numpy as np 
import pandas as pd  # data processing / CVS file I/O
from sklearn.model_selection import train_test_split
import csv

# file_path = '/Users/fionanicdao/loyola/machineLearning/hw4/marketing_campaign.csv'
# df = pd.read_csv(file_path, sep="\t")

file_path_xtrain = '/Users/fionanicdao/loyola/machineLearning/project/X_train.csv'
X_train = pd.read_csv(file_path_xtrain, sep="\t")

file_path_ytrain = '/Users/fionanicdao/loyola/machineLearning/project/y_train.csv'
y_train = pd.read_csv(file_path_ytrain, sep="\t")

file_path_xtest = '/Users/fionanicdao/loyola/machineLearning/project/X_test.csv'
X_test = pd.read_csv(file_path_xtest, sep="\t")

file_path_ytest = '/Users/fionanicdao/loyola/machineLearning/project/y_test.csv'
y_test = pd.read_csv(file_path_ytest, sep="\t")

""" MAKING FILES FOR TEST AND TRAIN
    # num_columns = df.shape[1]
# # print(num_columns)
# # print(df)
# y = df["Response"].values
# # print(y)
# X = df.iloc[0:,0:28].values
# print(X)

# X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=42)
# print(y_test)
# with open('X_test.csv','w',newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(X_test)
    
# with open('y_test.csv','w',newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(map(lambda x: [x], y_test))


# with open('y_train.csv','w',newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(map(lambda x: [x], y_train))
    
# with open('X_train.csv','w',newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows(X_train)
"""

