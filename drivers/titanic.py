#read in the data
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# Import titanic data 
path = '../rawData/titanic'
df = pd.read_csv(path + '/train.csv')
df.set_index('PassengerId', inplace=True)
#print(df.shape)
#print(list(df.columns))
#print(df.head(10))
# Visualise data:
print(df['Survived'].value_counts()) #survived (1) or not (0)
#sns.countplot(x='Survived', data=df)
#plt.show()
#plt.savefig('../results/count_plot.png')

# Feature engineering data:
# Create  boleans for each variable
listData = []
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
newDf = pd.concat(listData, axis=1)
print(newDf.head(10))
print(newDf.shape)
 
#newDf = df[['Survived','Pclass','Sex','Age', 'SibSp','Parch','Fare','Embarked']]
#newDf.dropna()
#print(newDf.loc[:,:])

# Logistic regression: Has the passenger survived (1) or not (0)?
from sklearn.linear_model import LogisticRegression
#s = df['Age']
#print(s)
m = LogisticRegression(C=1e10)
m.fit(newDf, y)
#X = df['Pclass']
#y = df['Survived']
#plt.scatter(X, y)
#plt.show()

# 
ypred = m.predict(newDf)
print(ypred)
print(y)
precison_score = []
from sklearn.metrics import precision_score
precison_score(y, ypred) #Note: true y first, after the y predicted.

#from sklearn.metrics import recall_score
#recall_score (y, ypred)

#from sklearn.metrics import confusion_matrix
#confusion_matrix (y, ypred)
# top_left (TN); bottom_right (TP); bottom_left (FN); top right (FP)
"""
TP : surviving passenger correctly predicted
TN : drowned passenger correctly predicted
FP : drowned passenger predicted as surviving
FN : surviving passenger predicted as drowned
"""



