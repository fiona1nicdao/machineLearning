"""
    Fiona Nicdao COMP 478
    Homework 4: Chapter 6 model selection
    PLEASE LOOK AT PDF FOR REPORT on SAKI 

    USED a new dataset:  Customer Personality Analysis
    Link: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis?resource=download

    PART 1 - 4  
"""
import numpy as np # type: ignore
import pandas as pd # type: ignore # data processing / CVS file I/O
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import LogisticRegression # type: ignore
from sklearn.metrics import f1_score # type: ignore

"""
part 1: randomly split the data into training (80%) and test (20%) sets
"""

# import marketing_campaign.csv file 
## please change the file_path to where you stored the dataset marketing_campaign.csv file 
file_path = '/Users/fionanicdao/loyola/machineLearning/hw4/marketing_campaign.csv'
df = pd.read_csv(file_path, sep="\t")
# print(df.info())

class MarketingCampaignDataset(object):
    def process_data(df):
        process_df = df
        process_df.isnull().sum(axis = 0)
        # there are 24 Nan values in the income column so replace the Nan with the median inccome
        
        median = process_df['Income'].median()
        process_df = process_df.fillna(median)
        # print(process_df.isnull().sum(axis = 0))
        # unique = process_df["Marital_Status"].unique()
        # print(unique)
        # education : Basic = 0, Graduation = 1, Master = 2, 2n Cycle = 2, PhD = 3
        process_df["Education"] = process_df["Education"].map({'Basic':0, 'Graduation':1,'Master':2, '2n Cycle':2, 'PhD':3})
        
        process_df["Marital_Status"] = process_df["Marital_Status"].map({'Single':1, 'Together':2, 'Married':3, 'Divorced':0, 'Widow':4, 'Alone':1, 'Absurd':1,'YOLO':1 })
        
        process_df.drop('Dt_Customer', axis=1, inplace=True)
        process_df.drop(['MntWines','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5'],axis=1)
        process_df["Response"] = process_df["Response"].map({0:-1,1:1})
        return process_df

       
        

df = MarketingCampaignDataset.process_data(df)
# print(df)
y = df["Response"].values
# print(y)
X = df.iloc[0:,0:28].values
# print(X)

X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=42)

# performance metric choosen is F1 score because of imbalance reponse (y) there are more 0 than 1 

"""
part 2: train and evaluate a classifer of your choice (e.g. logistic regression, SVM) using n-fold cross validation
"""
lr = LogisticRegression(C=0.1,solver='lbfgs', max_iter=5000)
lr.fit(X_train, y_train)
weights = lr.coef_
# intercept = lr.intercept_
print("part 1: LogisticRegression weights= ", weights) 
# print("part 1: LogisticRegression intercept= ",intercept)
# accuracy_train = lr.score(X_train, y_train)
# print("part 1: LogisticRegression accuracy = ", accuracy_train)
# y_pred = lr.predict(X_train)
# f1 = f1_score(y_train, y_pred, average="weighted")
# print("part 1: LogisticRegression f1 score: ", f1)

# n fold cross validation : try with n = 5 
# n =5;
# # divide the training data into 5 groups 
# num_examples = len(X_train)
# fold_size = num_examples // n 

# indices = np.random.permutation(num_examples)

# f1_scores = []

# for i in range(n):
#     start, end = i * fold_size, (i+1) * fold_size
#     test_indices = indices[start:end]
    
#     X_ntest, y_ntest = X_train[test_indices],y_train[test_indices]
#     train_indices = np.concatenate([indices[:start],indices[end:]])
#     X_ntrain, y_ntrain = X_train[train_indices],y_train[train_indices]
    
#     model = LogisticRegression(max_iter=200)
#     model.fit(X_ntrain, y_ntrain)
    
#     y_pred = model.predict(X_ntest)
#     accuracy = f1_score(y_ntest, y_pred)
#     f1_scores.append(accuracy)
#     print(f"Fold {i +1} accuracy : {accuracy:.4f}")
    
# mean_accuracy = np.mean(f1_scores)
# std_accuracy = np.std(f1_scores)
# print(f"\nMean accuracy: {mean_accuracy:.4f}") 
# print(f"Standard deviation of accuracy: {std_accuracy:.4f}")
def nfold (data, k) :
    fold_size = len(data) // k
    indices = np.arrange(len(data))
    folds = []
    for i in range(k) :
        test_indicies = indices[i+ fold_size : (i+1) * fold_size]
"""
part 3: implement your own grid search procedure from scrach which should include a search over at least two hyper-parameters
    C 
    penalty parameters
    
"""