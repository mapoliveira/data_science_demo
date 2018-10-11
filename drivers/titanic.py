######################################### Import and define styles
import sys
sys.path.insert(0, '../src')
import functionsTitanic

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

######################################### Function to calculate confusion matrix score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
def confusionMatrix_analysis(X, y, m):
    ypred = m.predict(X) # calculate ypred using model m
    print(confusion_matrix(y, ypred)) # show confusion table for both ytest with ypred
    """
    top_left (TN); bottom_right (TP); bottom_left (FN); top right (FP)
    TP : surviving passenger correctly predicted
    TN : drowned passenger correctly predicted
    FP : drowned passenger predicted as surviving
    FN : surviving passenger predicted as drowned
    """
    print("Precision score: " + str(precision_score(y, ypred))) #Note: ytest first, after the ypred
    print("Recall score: " + str(recall_score (y, ypred)))
    #ROC curve


######################################### Function to calculate bootstrapping
from sklearn.utils import resample
def bootstrapping_analysis(X, y, m):
    boots = []
    for i in range(1000):
        Xb, yb = resample(X, y)
        m.fit(Xb, yb)
        score = m.score(Xb, yb)
        boots.append(score)
        #print(i, score)

    # get percentiles for 90% confidence
    boots.sort()
    ci80 = boots[100:-100]
    print(f"80% confidence interval: {ci80[0]:5.2} -{ci80[-1]:5.2}")
    ci90 = boots[50:-50]
    print(f"90% confidence interval: {ci90[0]:5.2} -{ci90[-1]:5.2}")
    ci95 = boots[25:-25]
    print(f"95% confidence interval: {ci95[0]:5.2} -{ci95[-1]:5.2}")
    ci99 = boots[5:-5]
    print(f"99% confidence interval: {ci99[0]:5.2} -{ci99[-1]:5.2}")

######################################### Function to calculate bootstrapping
def bootCrossVal_analysis(Xtrain, Xtest, X, ytrain, ytest, y, m):
    # Bootstrap analysis:
    bootScoreTrain = bootstrapping_analysis(Xtrain, ytrain, m) # calculate bootstrapping score
    #bootScoreTest = bootstrapping_analysis(Xtest, ytest, m) # calculate bootstrapping score
    bootScoreAll = bootstrapping_analysis(X, y, m) # calculate bootstrapping score
    print("Bootstrap score (train data): " + str(bootScoreTrain))
    print("Bootstrap score (all data): " + str(bootScoreAll))
    
    # Cross-validation analysis:
    crossValScoreTrain = cross_val_score(X=Xtrain, y=ytrain, estimator=m, cv=5) 
    #crossValScoreTest = cross_val_score(X=Xtest, y=ytest, estimator=m, cv=5) 
    crossValScoreAll = cross_val_score(X=X, y=y, estimator=m, cv=5) 
    print("Cross-validation score (train data): " + str(crossValScoreTrain))
    print("Cross-validation score (all data): " + str(crossValScoreAll))  

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

#newDf = df[['Survived','Pclass','Sex','Age', 'SibSp','Parch','Fare','Embarked']]
#newDf.dropna()
#print(newDf.loc[:,:])
#print(df.iloc[:,6:11])

from functionsTitanic import featureEngineering
newDf = featureEngineering(df)

print(newDf.head(10))
numCategories = newDf.shape[0]
print("\nNumber of bolean categories based in all variables (except survival): "+ str(numCategories))

######################################### Split train/test:
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



