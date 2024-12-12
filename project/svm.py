"""
Created on Mon Dec 2 13:02:52 2024

@author: franciscogarzon


SVM Model


"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# load the dataset
data = pd.read_csv('/Users/franciscogarzon/Desktop/marketing_campaign.csv', delimiter="\t")

# Reading and cleaning the data
print("Initial data info:")
data.info()

# by drop rows with missing values
data = data.dropna() 

print(data.columns)


# feature engineering
data['Age'] = 2024 - data['Year_Birth']  # Getting the age
data['TotalKids'] = data['Kidhome'] + data['Teenhome']  # total number of children
data['EnrollmentYear'] = pd.to_datetime(data['Dt_Customer'], dayfirst=True).dt.year  # extracting enrollment year
data['Tenure'] = 2024 - data['EnrollmentYear']  # tenure

# dropping unnecessary columns
data = data.drop(['ID', 'Year_Birth', 'Dt_Customer', 'EnrollmentYear'], axis=1)

# selecting the most relevant features
features = data.drop('Response', axis=1)
target = data['Response']

# one-hot encode categorical variables
features = pd.get_dummies(features, drop_first=True)

# split the data into training and testing sets
# stratifying to maintain the class distribution
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42, stratify=target)

# scaling the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# initialize and train the SVM model
svm_model = SVC(kernel='rbf', class_weight="balanced", random_state=42)
svm_model.fit(x_train, y_train)

# make the predictions from the svm model
y_pred = svm_model.predict(x_test)

# evaluate the model with certain tests
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))