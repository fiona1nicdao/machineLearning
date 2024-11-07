"""
    Fiona Nicdao COMP 478
    Homework 4: Chapter 6 model selection
    PLEASE LOOK AT PDF FOR REPORT on SAKI 

    USED a new dataset:  Customer Personality Analysis
    Link: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis?resource=download

    PART 1 - 4  
"""
import numpy as np
import pandas as pd # data processing / CVS file I/O
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

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
        print(process_df.isnull().sum(axis = 0))

df_test = MarketingCampaignDataset.process_data(df)
y = df["Response"].values
# print(y)
X = df.iloc[0:,0:28].values
# print(X)

X_train, X_test,y_train, y_test = train_test_split(X,y, test_size=0.20,random_state=42)

# performance metric choosen is F1 score because of imbalance reponse (y) there are more 0 than 1 

"""
part 2: train and evaluate a classifer of your choice (e.g. logistic regression, SVM) using n-fold cross validation
"""
 