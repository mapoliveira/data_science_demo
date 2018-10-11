print('\n########## Complete pipeline for analysis of titanic dataset: ##########\n')

########## Import packages and define styles ##########
import pdb # python debugging
import sys
sys.path.insert(0, '../src') # Identify src directory
from functionsTitanic import * # Import functions specific to titanic dataset
from generalFunctions import * # Import costume made general functions (contains classification functions)
# Import required python packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
#import seaborn as sns
# Define plot styles
#plt.rc("font", size=14)
#sns.set(style="white")
#sns.set(style="whitegrid", color_codes=True)

########## Read Titanic data ##########
path = '../rawData/titanic'
df = pd.read_csv(path + '/train.csv')
df.set_index('PassengerId', inplace=True)
print("\nTitanic dataset loaded.")
print("\nDataset variables:")
print(list(df.columns))
numPassengers = df.shape[0] + 1
print("\nNumber of passengers: " + str(numPassengers))

########## Visualise and clean data ##########
print("\nSurvival count: ")
print(df['Survived'].value_counts()) #survived (1) or not (0)
#sns.countplot(x='Survived', data=df)
#plt.show()
#plt.savefig('../results/count_plot.png')

newDf = featureEngineering(df)
print(newDf.head(10))
numCategories = newDf.shape[0]
print("\nNumber of bolean categories considered (except survival): "+ str(numCategories))

########## Split train/test ##########
# Split dataset into train and test data
y = df['Survived']
X = newDf
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

########## Linear Regression (hyperparameters test)
n_estimators = [0.1, 1.0, 10.0, 100.0, 1000.0, 100000.0] # trees
depths = None
scoring = 'accuracy'
n_jobs = 2 
cv = 5
testMultipleHyperParameters(Xtrain, ytrain, 'LogisticRegression', n_estimators, depths, scoring, n_jobs, cv)

########## Decision tree (hyperparameters test)
#n_estimators = [2, 20, 40, 80, 100] # trees
#depths = [2, 3, 4, 5]
#scoring = 'accuracy'
#n_jobs = 2 
#cv = 5
#testMultipleHyperParameters(Xtrain, ytrain, 'RandomForest', n_estimators, depths, scoring, n_jobs, cv)

########## Random Forest (hyperparameters test)
n_estimators = [2, 20, 40, 80, 100] # trees
depths = [2, 3, 4, 5]
scoring = 'accuracy'
n_jobs = 2 
cv = 5
testMultipleHyperParameters(Xtrain, ytrain, 'RandomForest', n_estimators, depths, scoring, n_jobs, cv)


########## 1. Logistic classification method ##########
print("\n###  Classification method: Logistic regression  ###")
from sklearn.linear_model import LogisticRegression
## Logistic regression model:
m = LogisticRegression(C=1e5, solver='lbfgs', max_iter=1000)

## Analyse Train/Test data:
bootCrossVal_analysis(Xtrain, Xtest, X, ytrain, ytest, y, m)

## Fit model and calculate quality:
m.fit(Xtrain, ytrain) # fit model with the training dataset
logisticRegressionClassification_score = m.score(Xtrain, ytrain)

print("Logistic regression method score: " + str(m.score(Xtest, ytest)))
print("Confusion table analysis: ")
confusionMatrix_analysis(Xtrain, ytrain, m)

########## 2. Decision tree classification method ##########
print("\n##########Classification method: Decision tree")
from sklearn.tree import DecisionTreeClassifier
## Decision tree model:
m = DecisionTreeClassifier(max_depth=4)

## Analyse Train/Test data:
bootCrossVal_analysis(Xtrain, Xtest, X, ytrain, ytest, y, m)

## Fit model and calulate quality:
m.fit(Xtrain, ytrain) # fit the model with the training dataset
decisionTreeClassification_score = m.score(Xtrain, ytrain) # model quality

print("Decision tree classification method: " + str(decisionTreeClassification_score))
print("Confusion table analysis: ")
confusionMatrix_analysis(Xtrain, ytrain, m)

## Compare scores in Decision tree model:

########## 3. Random forest classification method ##########
print("\n##########Classification method: Random forest")
from sklearn.ensemble import RandomForestClassifier
## RandomForest-based model:
m = RandomForestClassifier(n_estimators=5, max_depth=2)
#n_estimators in the number of trees and max_depth is the number of levels in the tree (questions)

## Analyse Train/Test data:
bootCrossVal_analysis(Xtrain, Xtest, X, ytrain, ytest, y, m)

## Fit model and calculate quality:
m.fit(Xtrain, ytrain) # fit the model with the training dataset
randomForestTest_score = m.score(Xtrain, ytrain) # model quality

print("Randomforest method score: " + str(randomForestTest_score))
print("Confusion table analysis: ")
confusionMatrix_analysis(Xtrain, ytrain, m)

## Compare scores in RandomForest model:

print("\n### Has the passenger survived (1) or not (0)? ###")

