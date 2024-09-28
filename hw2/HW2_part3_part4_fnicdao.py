"""
  Fiona Nicdao  COMP 479
  Homework 2: Percepton algorithm and Adaline algorithm

    PLEASE LOOK AT WORD DOC FOR REPORT on Saki 

    PART 3: Titanic dataset (training and test) on AdalineGD
    PART 4: Most predictive features of your Titanic Model
"""
import numpy as np # type: ignore # linear algebra
import pandas as pd # type: ignore # data processing / CVS file I/O
import matplotlib.pyplot as plt # type: ignore # for scatter plot 
from Models import AdalineGD, TitanicDataset # my perceptron algorithm in Models.py 
    # and function for preprocessing Titanic Dataset

# import train.csv file 
## please change the file_path to where you store your train.csv file 
file_path = '/Users/fionanicdao/loyola/machineLearning/titanic_hw1/train.csv'
df = pd.read_csv(file_path)

# preprocess data : remove unused data and convert string values to numeric values
df = TitanicDataset.process_data(df)

#  shuffle the examples and split data : training 0.7 and test 0.3
df = df.sample(frac = 1)
ratio = 0.7 
total_rows = df.shape[0]
df_size = int(total_rows*ratio)
df_train = df[0:df_size]
df_test = df[df_size:]

"""
Training Dataset
"""
# set X(features) and y (target class labels) from dataset
y = df_train["Survived"].values
# X = df_train.iloc[0:,[1,5]].values
X = df_train.iloc[0:,1:5].values
# print(df_train["Survived"])
# train training dataset with Adaline model 
ada_gd = AdalineGD(n_iter=300, eta=0.0001)
ada_gd.fit(X, y)
# compare_y = pd.DataFrame({"pre": ada_gd.predict, "right": y})
compare_y = ada_gd.net_input(X)
# print(y)
# print(ada_gd.predict(X))
# print(compare_y)
score = np.mean(ada_gd.predict(X) == y)
print(score)
print("Training dataset's sum of squared error = " + str(ada_gd.cost_[-1]))
# plot Misclassification errors vs the number of epochs for training dataeset
plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.tight_layout()
# plt.show()

"""
Test Dataset 
"""
# set X(features) and y (target class labels) from dataset
y_test = df_test["Survived"].values
X_test = df_test.iloc[0:,1:5].values

# train testing dataset with Adaline model
ada_gd_test = AdalineGD(n_iter=1000, eta=0.0001)
ada_gd_test.fit(X_test, y_test)
print("Training dataset's sum of squared error = " + str(ada_gd_test.cost_[-1]))
# plot Misclassification errors vs the number of epochs for training dataeset
plt.plot(range(1, len(ada_gd.cost_) + 1), ada_gd.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.tight_layout()
# plt.show()

"""
PART 4:  Most predictive features of your Titanic Model
"""
# list of features 
features = list(df_train.columns)
# list of weights for each feature
weights = abs(ada_gd.w_)

# plot  the weights and features 
# remove the survival feature and weight 
bar_color= ['tab:red','tab:blue','tab:green','tab:purple']
plt.bar(features[1:],weights[1:],label=features[1:],color=bar_color)
plt.xlabel('features')
plt.ylabel('weights')
plt.legend(loc='upper left')
# plt.show()

print("\n Part 4: " +
      "\n Looking at the weights for each feature in order most" +
      "\n predictive features is Sex and Pclass")
