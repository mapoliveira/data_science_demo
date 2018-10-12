print('\n########## Complete pipeline for analysis of titanic dataset: ##########\n')

########## Import packages and define styles ##########
import pdb # python debugging
import sys
sys.path.insert(0, '../src') # Identify src directory
from titanicFunctions import * # Import functions specific to titanic dataset
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
df = pd.read_csv(path + '/all/train.csv')
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
#from sklearn.model_selection import train_test_split
#Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)
Xtrain = X
ytrain = y

########## Linear Regression (hyperparameters test)
n_estimators = [0.1, 1.0, 10.0, 100.0, 1000.0, 100000.0] # trees
depths = None
scoring = 'accuracy'
n_jobs = 4 
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
n_jobs = 4 
cv = 5
#RFbest_estimator_, RFbest_params_, RFbest_score_ = testMultipleHyperParameters(Xtrain, ytrain, 'RandomForest', n_estimators, depths, scoring, n_jobs, cv)
testMultipleHyperParameters(Xtrain, ytrain, 'RandomForest', n_estimators, depths, scoring, n_jobs, cv)

##########  Random forest classification method ##########
print("\n########## Best classification method: Random forest")
from sklearn.ensemble import RandomForestClassifier
n_estimators = 80
#n_estimators = RFbest_params_['n_estimators'] 
print(n_estimators)
max_depth = 5
#max_depth = RFbest_params_['max_depth']
print(max_depth)

m = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
#n_estimators in the number of trees and max_depth is the number of levels in the tree (questions)

## Analyse Train/Test data:
#bootCrossVal_analysis(Xtrain, Xtest, X, ytrain, ytest, y, m)

m.fit(Xtrain, ytrain) # fit the model with the training dataset
randomForestTest_score = m.score(Xtrain, ytrain) # model quality
print("\nRandomforest method score: " + str(randomForestTest_score))

##########  Confusion table analysis ##########
print("\n########## Confusion table analysis: ")
confusionMatrix_analysis(Xtrain, ytrain, m)

print("\n### Has the passenger survived (1) or not (0)? ###")
path = '../rawData/titanic'
pred = pd.read_csv(path + '/predict.csv')
pred.set_index('PassengerId', inplace=True)

Xpred = featureEngineering(pred)

# Predict
ypred = m.predict(Xpred)

# format CSV output
Xpred['Survived'] = ypred
Xpred.to_csv('result.csv') # submit to kaggle

########## 1. Logistic classification method ##########
#print("\n###  Classification method: Logistic regression  ###")
#from sklearn.linear_model import LogisticRegression
## Logistic regression model:
#m = LogisticRegression(C=1e5, solver='lbfgs', max_iter=1000)

## Analyse Train/Test data:
#bootCrossVal_analysis(Xtrain, Xtest, X, ytrain, ytest, y, m)

## Fit model and calculate quality:
#m.fit(Xtrain, ytrain) # fit model with the training dataset
#logisticRegressionClassification_score = m.score(Xtrain, ytrain)

#print("Logistic regression method score: " + str(m.score(Xtest, ytest)))

#print("Confusion table analysis: ")
#confusionMatrix_analysis(Xtrain, ytrain, m)

########## 2. Decision tree classification method ##########
#print("\n##########Classification method: Decision tree")
#from sklearn.tree import DecisionTreeClassifier
## Decision tree model:
#m = DecisionTreeClassifier(max_depth=4)

## Analyse Train/Test data:
#bootCrossVal_analysis(Xtrain, Xtest, X, ytrain, ytest, y, m)

## Fit model and calulate quality:
#m.fit(Xtrain, ytrain) # fit the model with the training dataset
#decisionTreeClassification_score = m.score(Xtrain, ytrain) # model quality

#print("Decision tree classification method: " + str(decisionTreeClassification_score))
#print("Confusion table analysis: ")
#confusionMatrix_analysis(Xtrain, ytrain, m)

## Compare scores in Decision tree model:


