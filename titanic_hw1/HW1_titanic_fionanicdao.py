"""
  Fiona Nicdao
  COMP 479
  Homework 1: Titanic dataset from Kaggle.com :
    https://www.kaggle.com/c/titanic

  Objective: your goal is to develop an algorithm to make a prediction for each
            passenger whether that passenger will survive the sinking

            Can use: python, NumPy, pandas, matplotlib etc.
            CANNOT use: scikit-learn and machine learning libraries

            PLEASE LOOK AT WORD DOC FOR REPORT on Saki 
"""

# import numpy as np # linear algebra
import pandas as pd # data processing / CVS file I/O

# import train.csv file 
## please change the file_path to where you store your train.csv file 
file_path = '/Users/fionanicdao/loyola/machineLearning/titanic_hw1/train.csv'
df = pd.read_csv(file_path)

# import test.csv file
## please change the file_path to where you store your test.csv file 
file_path = '/Users/fionanicdao/loyola/machineLearning/titanic_hw1/test.csv'
df_test = pd.read_csv(file_path)

def analyze_survivors(df) :
  # Total number of passengers 
  row_count = len(df)
  print("total rows = " + str(row_count))
  # correct number of survivors 
  num_survivors = df['Survived'].value_counts()[1]
  print("survived = " + str(num_survivors) )
  # correct number of perished passengers 
  num_dead = df['Survived'].value_counts()[0]
  print("dead = " + str(num_dead))
  # dataframe of only survivors 
  df_survivors = df.drop(df[df['Survived'] == 0].index)
  print(df_survivors)

  # look at data and count how many are Pclass A proxy for socio-economic status (SES)
  print("1st class = " +  str(df_survivors['Pclass'].value_counts()[1]))
  print("2nd class = " + str(df_survivors['Pclass'].value_counts()[2]))
  print("3nd class = " + str(df_survivors['Pclass'].value_counts()[3]))
  nAn_pclass = df_survivors['Pclass'].isna().sum()
  print("NaN for pcalss " + str(nAn_pclass))

  # look at data and count how many are sex
  print("males = " + str(df_survivors['Sex'].value_counts()['male']))
  print("female = " + str(df_survivors['Sex'].value_counts()['female']))
  nAn_Sex = df_survivors['Sex'].isna().sum()
  print("NaN for sex " + str(nAn_Sex))

  # look at data and count how many are age
  young = df_survivors['Age'].between(0,30, inclusive='both').sum()
  print("survivors under or at 30yrs = " + str(young))
  old = df_survivors['Age'].between(31,100, inclusive='both').sum()
  print("survivors over 30yrs = " + str(old))
  median = df_survivors['Age'].median()
  print("median age = " + str(median))
  mean = df_survivors['Age'].mean()
  print("mean age = " + str(mean))
  nAn_Age = df_survivors['Age'].isna().sum()
  print("NaN for  " + str(nAn_Age))

  # sibsp: # of siblings / spouses aboard the Titanic
  no_sibs = df_survivors['SibSp'].value_counts()[0]
  print("no sibs = " + str(no_sibs))
  one_sibs = df_survivors['SibSp'].value_counts()[1]
  print("one sibs = " + str(one_sibs))
  nAn_sibs = df_survivors['SibSp'].isna().sum()
  print("NaN sibs = " + str(nAn_sibs))

def survivors(df) :
  # copy of dataframe to remove passengers predicted to perish at the titanic
  df_predicted_survivors = df

  # removing 2nd class passengers 
  Pclass_2 = df['Pclass'] == 2
  df_predicted_survivors = df_predicted_survivors[~Pclass_2]

  # removing male passengers 
  sex_male = df['Sex'] == 'male'
  df_predicted_survivors = df_predicted_survivors[~sex_male]
  
  # removing passengers with 2 or more siblings / spouses
  more_than_2_sibsp = df['SibSp'] >= 2
  df_predicted_survivors = df_predicted_survivors[~more_than_2_sibsp]

  # count the rows, passengers predcited to survive the titanic accident 
  predicted_survivors = len(df_predicted_survivors)
  
  return(df_predicted_survivors, predicted_survivors)


def compute_accuracy_of_survivors(df, df_predicted_survivors, predicted_survivors) :

  # count the number of correct predicted survivors 
  num_survivors = df_predicted_survivors['Survived'].value_counts()[1]

  # count the number of incorrect predicted survivors 
  num_dead = df_predicted_survivors['Survived'].value_counts()[0]

  # correct number of survivors from original training data 
  correct_num_survivors = df['Survived'].value_counts()[1]

  # accuracy of predictions of surviviors 
  accuracy = (num_survivors /correct_num_survivors) * 100

  print("\n\n My algoritm predicted " + str(predicted_survivors) + " passengers survived. \n However, " 
        + str(num_dead) + " passengers predicted to survived actually perished. \n So only " 
        + str(num_survivors) + " passengers are correctly labeled of the " + str(correct_num_survivors) 
        + " correct number \n of surivied passengers from the training data")
  print("\n\nThe accurancy of my algorithm to predict survivors from the "+
        "titanicdataset is " + str(accuracy) +" %  \n\n")

# analyze_survivors(df) # just to show my work for analyzing the train.cvs data 

# predicted survivors using training data 
df_predicted_survivors, predicted_survivors = survivors(df) 

# accuracy of 'survivors' algorithm 
compute_accuracy_of_survivors(df, df_predicted_survivors, predicted_survivors) 

# predicted survivors using test data 
df_predicted_survivors, predicted_survivors,  = survivors(df_test)
print("\n\n My algorithm predicts " + str(predicted_survivors) + 
      " passengers will survive from the test data")
