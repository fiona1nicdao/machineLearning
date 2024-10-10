"""
    Fiona Nicdao COMP 479 
    Homework 3 : Logistic Regression / SVM classifers and k-nearst neighbors 
                 classifer (KNN) 
    PLEASE LOOK AT PDF FOR REPORT on SAKI 
    
    Part 1: use Logistic regression from scikit - learn 
"""
import numpy as np  # linear algebra
import pandas as pd # data processing / CVS file I/O
# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

#import linear_dataset.csv file 
## please change the file_path to where you store linear_dataset.csv file
file_path = '/Users/fionanicdao/loyola/machineLearning/titanic_hw1/train.csv'
df = pd.read_csv(file_path)
# df_training (70%)
# df_development (15%)
# df_test (15%)
# part 1: use logistic regression
class TitanicDataset(object):
    def process_data(df):
        # drop useless data : PassengerId , Ticket , Fare , Cabin , Embarked
        df = df.drop(["PassengerId", "Ticket", "Fare", "Cabin", "Embarked","Name","Age"], axis=1)

        df["Survived"] = df["Survived"].map({0 : -1, 1:1})
        # no processing done with Pclass
         
        # sex : male = 0 and female = 1
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

        # parch 
        def parch_separate(passenger):
            parch = passenger
            if( parch > 0) : # if passenger have a parent / children  = 1
                return 1
            else : # else passenger has no parent / children = 0 
                return 0
            
        df["Parch"] = df["Parch"].apply(parch_separate)

        # sibsp
        def sibsp_separate(passenger):
            sibsp = passenger
            if(sibsp > 0): # if passengers has a sibling / spouse = 1
                return 1
            else: # passenger has no parent / children = 0
                return 0

        df["SibSp"] = df["SibSp"].apply(sibsp_separate)

        return df
    
df = TitanicDataset.process_data(df)
# print(df)
# set X(features) and y (target class labels) from dataset
y = df["Survived"].values
# print(y)
# X = df_train.iloc[0:,[1,5]].values
X = df.iloc[0:,1:5].values
# print(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=42)
# print(y_train)
X_test, X_develop, y_test, y_develop = train_test_split(X_test,y_test, test_size=0.50, random_state=42)
# print(y_develop)
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train, y_train)
# do F1 for accurarcu 

# print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))