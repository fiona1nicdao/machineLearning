"""
    Fiona Nicdao COMP 479 
    Homework 3 : Logistic Regression and k-nearst neighbors 
                 classifer (KNN) 
    PLEASE LOOK AT PDF FOR REPORT on SAKI 
    
    Part 1 - 5
"""
import numpy as np  # linear algebra
import pandas as pd # data processing / CVS file I/O
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
from collections import Counter

#import Titanic train.csv file 
## please change the file_path to where you store titanic train.csv file
file_path = '/Users/fionanicdao/loyola/machineLearning/titanic_hw1/train.csv'
df = pd.read_csv(file_path)

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
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.15, 
                                                    random_state=42)
X_train, X_develop, y_train, y_develop = train_test_split(X_train,y_train, 
                                                        test_size=0.16, 
                                                        random_state=42)

"""
part 1: use logistic regression from scikit-learn to train a classifier. 
use the default classifier hyperparameters. Evaluate your classifier on 
the development set. 
"""
lr = LogisticRegression()
lr.fit(X_train, y_train)
weights = lr.coef_
intercept = lr.intercept_
print("part 1: LogisticRegression weights= ", weights)
print("part 1: LogisticRegression intercept= ",intercept)
accuracy_train = lr.score(X_develop, y_develop)
print("part 1: LogisticRegression accuracy = ", accuracy_train)
y_pred = lr.predict(X_develop)
f1 = f1_score(y_develop, y_pred, average="weighted")
print("part 1: LogisticRegression f1 score: ", f1)

"""
part 2 : now explre the classifier hyperparameters and see if you can
improve your model's performance on the developement set

Hyperparameters tested: C value, random state, solvers
"""
# C values : 0.0001, 0.001, 0.01, 0.1, 1,10, 100, 1000 
accuracy_score = []
Cs = [0.0001, 0.001, 0.01, 0.1, 1,10, 100, 1000]
C_num = [1,2,3,4,5,6,7,8]
for c in Cs:
    # print(c)
    lr_c = LogisticRegression(C=c)
    lr_c.fit(X_train, y_train)
    acc = lr_c.score(X_develop, y_develop)
    accuracy_score.append(acc)

fig, ax = plt.subplots()
ax.plot(C_num, accuracy_score)
ax.set_xticklabels([0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000])
ax.set(xlabel="C",
       ylabel="Accuracy",
       title="Performance of Logistic Regression with Change of C values ")
plt.show()

# random state a range from 1 to 50 
accuracy_rs = []
r_states = range(1,50)
for r in r_states :
    lr_randomstate = LogisticRegression(random_state=r).fit(X_train, y_train)
    acc = lr_randomstate.score(X_develop, y_develop)
    accuracy_rs.append(acc)
fig, ax = plt.subplots()
ax.plot(r_states, accuracy_rs)
ax.set(xlabel="random state",
       ylabel="Accuracy",
       title="Performance of Logistic Regression with Change of Random State")
plt.show()

# solver algorithm :'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky','sag', 'saga'
accuracy_solver = []
solvers = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky','sag', 'saga']
for s in solvers :
    lr_solver = LogisticRegression(solver=s).fit(X_train, y_train)
    acc = lr_solver.score(X_develop, y_develop)
    accuracy_solver.append(acc)
fig, ax = plt.subplots()
ax.plot(solvers, accuracy_solver)
ax.set(xlabel="solvers",
       ylabel="Accuracy",
       title="Performance of Logistic Regression Solver Algorithm")
plt.show()

# penalty = {‘l1’, ‘l2’, ‘elasticnet’, None}, default=’l2’

"""
part 3 : implement your own k-nearest neighbors classifer (KNN) from scratch. 
Tune the k value to achive the best possible performance on the development set.
 the KNN algorithm can be summarized by the following steps: 
 1. choose the number of k and a distance metric
 2. find the k-nearst neighbors of the data record that we want to classify
 3. assign the class label by majority vote 
"""

class KNNClassifier:
    def __init__(self, k=10):
        self.k = k

    def fit(self, X_train, y_train):
        # store the training data
        self.X_train = X_train
        self.y_train = y_train

    def distance(self, x1, x2):
        # compute the distance between two points
        return np.sqrt(np.sum((x1-x2) ** 2))
    
    def predict(self, X_test):
        # predict the class for each example in X_test
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
    
    def _predict(self, x):
        #predict for class for one single example x
        #compute the distances to all training samples
        distance_x = [self.distance(x,x_train) for x_train in self.X_train]
        # get the indices of the k nearest neighbors
        k_indices = np.argsort(distance_x)[:self.k]
        #get the labels of the k nearest neighbors 
        k_nearest_label = [self.y_train[i] for i in k_indices]
        # return the most common label
        # most_common = counter_elements(k_nearest_label).most_common() 
        most_common = Counter(k_nearest_label).most_common(1)
        return most_common[0][0]   

    def accuracy(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = sum(y_pred == y_test) / len(y_test)
        return acc


knn = KNNClassifier()
knn.fit(X_train, y_train)
evaluation = knn.accuracy(X_develop, y_develop)
print("Part 2: KNN accuracy = ", evaluation)
y_predict = knn.predict(X_develop)
f1_knn = f1_score(y_develop, y_predict, average="weighted")
print("Part 2: KNN f1 score =", f1_knn)

#  test knn model acorss varying ks 
acuracies = []
ks = range(1,20)
for k in ks :
    knn = KNNClassifier(k=k)
    knn.fit(X_train, y_train)
    accuracy = knn.accuracy(X_develop, y_develop)
    acuracies.append(accuracy)

# Visualize accuracy vs. k
fig, ax = plt.subplots()
ax.plot(ks, acuracies)
ax.set(xlabel="k",
       ylabel="Accuracy",
       title="Performance of knn")
plt.show()

"""
part 4 : to establish the performance of a baseline system apply the Dummy-Classifer
from scikit-learn (sklearn.dummy.DummyClassifier) to your data. At minimum, try the 
following two values for the 'strategy' 
parameter:"most_frequent", "prior", "stratified", "uniform"
"""
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

"""
part 5: compare your best model that you built in step(2) to your best KNN model by 
evaluating them on the 'test' set. Document your results and include the performance 
of your baseline systems from step(4) in  your analysis for comparison 

Logistic Regression : 0.1, random state=1, solver='lbfgs'
KNN : k = 5 
"""
lr = LogisticRegression(C=1, random_state=1, solver='lbfgs')
lr.fit(X_train, y_train)
accuracy = lr.score(X_test, y_test)
print("Part 5: Logistic Regression accuracy = ", accuracy)
y_pred = lr.predict(X_test)
f1 = f1_score(y_test, y_pred, average="weighted")
print("Part 5: Logistic Regression F1 score = ", f1)


knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)
evaluation = knn.accuracy(X_test, y_test)
print("Part 5: KNN accuracy = ", evaluation)
y_predict = knn.predict(X_test)
f1_knn = f1_score(y_test, y_predict, average="weighted")
print("Part 5: KNN f1 score =", f1_knn)