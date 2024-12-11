
import tensorflow as tf
from tensorflow import keras
import numpy as np 
import pandas as pd  # data processing / CVS file I/O
# from numpy import array
# from tensorflow import keras
# from sklearn.metrics import f1_score
# from tensorflow.keras.utils import to_categorical
# from sklearn.pipeline import Pipeline

metric = keras.metrics.F1Score(average="weighted", threshold=0.5)
y_true = np.array([[1, 1, 1],
                   [1, 0, 0],
                   [1, 1, 0]], np.int32)
y_pred = np.array([[0.2, 0.6, 0.7],
                   [0.2, 0.6, 0.6],
                   [0.6, 0.8, 0.0]], np.float32)
metric.update_state(y_true, y_pred)
result = metric.result()
print(result)
# file_path_xtrain = '/Users/fionanicdao/loyola/machineLearning/project/X_train.csv'
# X_train = pd.read_csv(file_path_xtrain)

# file_path_ytrain = '/Users/fionanicdao/loyola/machineLearning/project/y_train.csv'
# y_train = pd.read_csv(file_path_ytrain)

# # X_train, X_validate,y_train, y_validate = train_test_split(X_train,y_train, test_size=0.10,random_state=42)
# print(y_train) 

# file_path_xtest = '/Users/fionanicdao/loyola/machineLearning/project/X_test.csv'
# X_test = pd.read_csv(file_path_xtest)

# file_path_ytest = '/Users/fionanicdao/loyola/machineLearning/project/y_test.csv'
# y_test = pd.read_csv(file_path_ytest)

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
file_path = '/Users/fionanicdao/loyola/machineLearning/hw4/marketing_campaign.csv'
data = pd.read_csv(file_path, sep="\t")

# cat_columns = X_train.select_dtypes(include=['object']).columns
# num_columns = X_train.select_dtypes(include=['int64','float64']).columns

# input_dim = X_train.shape[1]

# # Define the neural network model
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
# ])

# print("accurarcy is ", accuracy)
# print("f1 score: ", f1_score)
# write neural network 
# y_pred_prob = model.predict(X_test)

# y_test = np.array(y_test, np.int32)
# print(y_pred)
# print(np.concatenate(y_pred))
# print(y_test)

# metric = keras.metrics.F1Score(average="weighted", threshold=0.5)
# metric.update_state(y_test,y_pred)
# f1 = metric.result()

# tf.keras.metrics.F1Score(
#     average=None, threshold=None, name='f1_score', dtype=None
# )
# # Predicting on the test set

# y_pred = (y_pred_prob > 0.5).astype(int)
# y_pred_prob = model.predict(X_test)
# y_pred = np.concatenate(y_pred)
# f1 = f1_score(y_test, y_pred)
# f1 = f1_score(y_test,y_pred)
# Calculating F1 score
print("F1 Score:", f1)