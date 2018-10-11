######################################### Import and define styles
import sys
sys.path.insert(0, '../src') # identify src directory
import functionsTitanic # import functions specific to titanic dataset
import generalFunctions

import pdb
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
## Import classification methods
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

######################################### Read Titanic data:
print("\n### Titanic dataset ###")
path = '../rawData/titanic'
df = pd.read_csv(path + '/train.csv')
df.set_index('PassengerId', inplace=True)
print("\nVariables in the dataset:")
print(list(df.columns))
numPassengers = df.shape[0] + 1
print("\nNumber of passengers: " + str(numPassengers))

######################################### Visualise and clean data:
print("\nSurvival count: ")
print(df['Survived'].value_counts()) #survived (1) or not (0)
#sns.countplot(x='Survived', data=df)
#plt.show()
#plt.savefig('../results/count_plot.png')

from functionsTitanic import featureEngineering
newDf = featureEngineering(df)

#newDf = df[['Survived','Pclass','Sex','Age', 'SibSp','Parch','Fare','Embarked']]
#newDf.dropna()
#print(newDf.loc[:,:])
#print(df.iloc[:,6:11])

print(newDf.head(10))
numCategories = newDf.shape[0]
print("\nNumber of bolean categories based in all variables (except survival): "+ str(numCategories))

######################################### Split train/test:
from generalFunctions import *


print("\n### Has the passenger survived (1) or not (0)? ###")
# Split dataset into train and test data
y = df['Survived']
X = newDf
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

######################################### 1. Logistic classification method: 
print("\n##########Classification method: Logistic regression")
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

######################################### 2. Decision tree classification method:
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

######################################### 3. Random forest classification method:
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



