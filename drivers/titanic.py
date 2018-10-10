### Import and define styles
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
#from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
## Import classification methods
from sklearn.linear_model import LogisticRegression

### Function to calculate confusion matrix score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
def confusionMatrix_analysis(X, y, m)
    ypred = m.predict(X) # calculate ypred using model m
    confusion_matrix(y, ypred) # show confusion table for both ytest with ypred
    """
    top_left (TN); bottom_right (TP); bottom_left (FN); top right (FP)
    TP : surviving passenger correctly predicted
    TN : drowned passenger correctly predicted
    FP : drowned passenger predicted as surviving
    FN : surviving passenger predicted as drowned
    """
    print("Precision score: " + str(precison_score(y, ypred))) #Note: ytest first, after the ypred
    print("Recall score: " + str(recall_score (y, ypred)))

### Read Titanic data
print("\n### Titanic dataset ###")
path = '../rawData/titanic'
df = pd.read_csv(path + '/train.csv')
df.set_index('PassengerId', inplace=True)
print("\nVariables in the dataset:")
print(list(df.columns))
numPassengers = df.shape[0] + 1
print("\nNumber of passengers: " + str(numPassengers))

### Visualise and clean data: ###
print("\nSurvival count: ")
print(df['Survived'].value_counts()) #survived (1) or not (0)
#sns.countplot(x='Survived', data=df)
#plt.show()
#plt.savefig('../results/count_plot.png')

#newDf = df[['Survived','Pclass','Sex','Age', 'SibSp','Parch','Fare','Embarked']]
#newDf.dropna()
#print(newDf.loc[:,:])

### Split train/test: ###
print("\n### Has the passenger survived (1) or not (0)? ###")
# Split dataset into train and test data
y = df['Survived']
X = df[df.columns[1 :]]

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)

### Feature engineering data: ###
print("Creating boleans categories for each variable...")
listData = [] #List of data frames
for i in df.columns:
    if i == 'Survived':
        y = df[i]
        #print(df[i].dtype)
    elif i == 'Name' or i == 'Ticket' or i == 'Cabin':
        print(i)
    elif df[i].dtype == 'float64':
        #print(i)
        dataBins = pd.cut(df[i], bins = 3, labels = [1,2,3])
        #print(df[i].dtype)
    else:   
        #print(df[i].dtype)
        listData.append(pd.get_dummies(df[i]))

newDf = pd.concat(listData, axis=1) # Concatenate data frames
print(newDf.head(10))
numCategories = newDf.shape[0]
print("\nNumber of bolean categories based in all variables (except survival): "+ str(numCategories))

### Build model: ###

print("\n### Confusion table analysis: ###")
cnf_matrix = confusion_matrix(y_test, y_pred)
m.fit(Xtrain, ytrain) #train model
ypred = m.predict(newDf)

### Logistic classification method: 
from sklearn.linear_model import LogisticRegression
m = LogisticRegression(C=1e10)
m.fit(Xtrain, ytrain) # train model
print("Logistic regression method score: " + str(m.score(Xtest, ytest)))



