""" 
    @author : Fiona Nicdao
    Model   : neural network  
"""
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
import pandas as pd  # data processing / CVS file I/O
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score, classification_report
import csv

file_path = '/Users/fionanicdao/loyola/machineLearning/hw4/marketing_campaign.csv'
df = pd.read_csv(file_path, sep="\t")
# Identify features and target
target = 'Response'

# separate categorical and numerical columns 
categorical_features = ['Education', 'Marital_Status']
numerical_features = ['Year_Birth', 'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits',
                      'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
                      'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases',
                      'NumWebVisitsMonth']

# Drop unnecessary columns for simplicity
df = df.drop(columns=['ID', 'Dt_Customer', 'Z_CostContact', 'Z_Revenue'])

# Handle missing values by imputing with median for numerical columns
df['Income'].fillna(df['Income'].median(), inplace=True)

#process data even more  
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(),numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'),categorical_features)
    ]
)

# Split data into features and target
X = df[categorical_features + numerical_features]
y = df[target]
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

#build and train neural network 
input_dim = X_train.shape[1]
model = Sequential([
    Dense(16, activation = 'relu', input_dim=input_dim),
    Dense(16, activation = 'relu'),
    Dense(8, activation = 'relu'),
    Dense(1, activation = 'sigmoid') # Binary Classification 
])
# compile 
model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy', 'precision', 'recall'])
#train the model 
history = model.fit(X_train,y_train, validation_data=(X_test, y_test),epochs=20, batch_size=32, verbose=1)

# accuracy vs f1 scoring 
loss, accuracy, precision, recall= model.evaluate(X_test,y_test,verbose=0)
print("Loss: ", loss)

y_pred = np.concatenate((model.predict(X_test)) > 0.5).astype(int)
y_test = y_test.to_numpy()

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
