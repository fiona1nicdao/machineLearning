"""
    Fiona Nicdao COMP 479 
    Homework 3 : Logistic Regression / SVM classifers and k-nearst neighbors 
                 classifer (KNN) 
    PLEASE LOOK AT PDF FOR REPORT on SAKI 
    
    Part 1: use Logistic regression from scikit - learn 
"""
import numpy as np  # linear algebra
import pandas as pd # data processing / CVS file I/O

#import linear_dataset.csv file 
## please change the file_path to where you store linear_dataset.csv file
file_path = '/Users/fionanicdao/loyola/machineLearning/hw2/linear_dataset.csv'
df = pd.read_csv(file_path)