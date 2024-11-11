"""
    Fiona Nicdao COMP 478
    Homework 4: Chapter 6 model selection
    PLEASE LOOK AT PDF FOR REPORT on SAKI 

    used Titanic dataset

    PART 1 - 4  
"""
import numpy as np 
import pandas as pd  # data processing / CVS file I/O
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import f1_score 

"""
PART 1: Split Training (80%) and Test (20%)
"""
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
        # Get the counts of each unique value && # Calculate the percentage for each value
        counts = df['Survived'].value_counts()
        percentages = (counts / counts.sum()) * 100 
        # print(percentages)
        
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
# df_training (80%), df_test (20%)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42)

"""
PART 2: train and evaluate a classifer of your choice (e.g. logistic regression, SVM)
        using n-fold cross validation
"""
print("### part 2 ####")
def n_fold_cross_validation(X, y, n=5, C=1.0, penalty='l2', solver='lbfgs') : 
    # n fold cross validation : default is  n = 5 
    # divide the training data into 5 groups 
    num_examples = len(X)
    fold_size = num_examples // n 

    indices = np.random.permutation(num_examples)

    f1_scores = []

    for i in range(n):
        start, end = i * fold_size, (i+1) * fold_size
        test_indices = indices[start:end]
        
        X_ntest, y_ntest = X[test_indices],y[test_indices]
        train_indices = np.concatenate([indices[:start],indices[end:]])
        X_ntrain, y_ntrain = X[train_indices],y[train_indices]
        
        lr = LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=1000)
        lr.fit(X_ntrain, y_ntrain)
        
        y_pred = lr.predict(X_ntest)
        f1 = float(f1_score(y_ntest, y_pred))
        f1_scores.append(f1)
        print(f"Fold {i +1} f1 : {f1:.4f}")
        
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    print(f"\nMean f1: {mean_f1:.4f}") 
    print(f"Standard deviation of f1: {std_f1:.4f}")
    
    return(mean_f1, C, penalty, lr)

n_fold_cross_validation(X_train,y_train,10)
"""
PART 3: implement your own grid search procedure from scratch which should include
        a search over at least two hyper-parameters. run a grid search over these 
        hyperparameters and decide which hyperparameter combination gives you the 
        best performance your grid search should rely on your n-fold cross validation
        implementation from subproblem 3 above the score different models 
"""
print("\n### part 3 ###")
def grid_search_C_and_penalty(X,y) :
    C_values = [0.01, 0.1, 1, 10, 100]
    penalty_values = ['l1', 'l2']
    
    best_params = None
    best_f1_score = 0
    results = []
    
    for C in C_values :
        for penatly in penalty_values : 
            if penatly == 'l1' : 
                solver = 'liblinear' # supports l1 regularization
            else : solver = 'lbfgs' # supports l2 regularization
            
            f1, C, penatly, lr = n_fold_cross_validation(X,y,10,C, penatly,solver)
            
            results.append({'C': C, 'penalty': penatly, 'f1':f1})
            
            if f1 > best_f1_score :
                best_f1_score = f1
                best_params = {'C': C, 'penalty':penatly, 'solver':solver}
    # Display the results
    print("\n\n")
    print("Best Parameters:", best_params)
    print("Best Validation F1 Score:", best_f1_score)
    print("\nAll Results:")
    for result in results:
        print(result)
        
    return(best_params['C'], best_params['penalty'], best_params['solver'],best_f1_score)

best_C, best_penatly,best_solver, best_f1 = grid_search_C_and_penalty(X_train, y_train)

"""
PART 4 : evaluate the best model that you identify in part 3 and report its performance 
        on the test set. Compare this number to the performance of your best model on the 
        training set(i.e. train and test on the training data) and explain the difference 
"""

print("\n### part 4 ####")
print("The best C : ",best_C , "\nThe best penalty: ",best_penatly, "\nThe bet solver: ", best_solver )
print("The best f1 score: ", best_f1)

lr = LogisticRegression(C=best_C, penalty=best_penatly, solver=best_solver, max_iter=1000)
lr.fit(X_test,y_test)
            
y_pred = lr.predict(X_test)
f1_test = f1_score(y_test, y_pred)
print("test f1 score",f1_test)