"""
    Fiona Nicdao COMP 478
    Homework 4: Chapter 6 model selection
    PLEASE LOOK AT PDF FOR REPORT on SAKI 

    used Titanic dataset:  
    Link: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis?resource=download

    PART 1 - 4  
"""
import numpy as np # type: ignore
import pandas as pd # type: ignore # data processing / CVS file I/O
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import f1_score # type: ignore

#import Titanic train.csv file 
## please change the file_path to where you store titanic train.csv file
file_path = '/Users/fionanicdao/loyola/machineLearning/titanic_hw1/train.csv'
df = pd.read_csv(file_path)
#prepressing the Titanic Dataset 
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
# set X(features) and y (target class labels) from dataset
y = df["Survived"].values
X = df.iloc[0:,1:5].values
# df_training (70%), df_development (15%), df_test (15%)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)

"""
part 2: train and evaluate a classifer of your choice (e.g. logistic regression, SVM) using n-fold cross validation
"""
# lr = LogisticRegression()
# lr.fit(X_train, y_train)
# weights = lr.coef_
# intercept = lr.intercept_
# print("part 1: LogisticRegression weights= ", weights)
# print("part 1: LogisticRegression intercept= ",intercept)
# accuracy_train = lr.score(X_test, y_test)
# print("part 1: LogisticRegression accuracy = ", accuracy_train)
# y_pred = lr.predict(X_test)
# f1 = f1_score(y_test, y_pred, average="weighted")
# print("part 1: LogisticRegression f1 score: ", f1)

# n fold cross validation : try with n = 5 
n =10;
# divide the training data into 5 groups 
num_examples = len(X_train)
fold_size = num_examples // n 

indices = np.random.permutation(num_examples)

f1_scores = []

for i in range(n):
    start, end = i * fold_size, (i+1) * fold_size
    test_indices = indices[start:end]
    
    X_ntest, y_ntest = X_train[test_indices],y_train[test_indices]
    train_indices = np.concatenate([indices[:start],indices[end:]])
    X_ntrain, y_ntrain = X_train[train_indices],y_train[train_indices]
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_ntrain, y_ntrain)
    
    y_pred = model.predict(X_ntest)
    accuracy = f1_score(y_ntest, y_pred)
    f1_scores.append(accuracy)
    print(f"Fold {i +1} accuracy : {accuracy:.4f}")
    
mean_accuracy = np.mean(f1_scores)
std_accuracy = np.std(f1_scores)
print(f"\nMean accuracy: {mean_accuracy:.4f}") 
print(f"Standard deviation of accuracy: {std_accuracy:.4f}")

