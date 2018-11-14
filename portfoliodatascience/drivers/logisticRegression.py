# apply logistic regression
x1 = [-1.0, -.75,-.5,-.25,0.0,.25,.5,.75,1.0]
x2 = [abs(i) for i in x1]
print(x2)
from sklearn.linear_model import LogisticRegression
m= LogisticRegression()
import numpy as np
y = [1,1,1,0,0,0,1,1,1,] 
X1 =  np.array(x1).reshape((-1,1))
X2 = np.array([x1, x2]).T

result1 = m.fit(X1,y).score(X1,y)
result2 = m.fit(X2,y).score(X2,y)
print(result1)
print(result2)

print('Studying Feature Engineering: missing values') 
import pandas as pd
x = pd.Series([1.0, np.NaN, 2.0, 3.0, np.NaN, np.NaN, 6.0, 1.0, 1.0])
print(x)

#x.fillna(x.mean()) #replace NaN with mean
#x.fillna(0.0) #replace NaN with 0.0
#x.fillna(method = 'bfill') #NaN is replace with the next data point (with previous value 'ffill')
#x.dropna() #remove NaN
x.interpolate('linear') #interpolate NaN with 'linear' ('quadratic') regression between the neighbouring points
print(x)

print('Studying Feature Engineering: normalization')
#np.log(x) #logaritmic normalization
#np/ totalNum #normalizing using the maximum value of all events

#Re-scalling data:
#(x-min(x)/(max(x)-min(x))) #all numbers between 0 and 1
from sklearn.preprocessing import MinMaxScaler
x.values.reshape(-1,1)
MinMaxScaler(copy=True, feature_range=(0,1))
m.transform(X)

#Standard scalling: It does not change the proporcions of the data but it will stretch the data to fit into a known mean and std. Less sensitive to outliers and values will have min and max different than [0,1]. Assumes that data is normaly distributed.
from sklearn.preprocessing import StandardScaler
m = StandardScaler()
m.transform(X)

#Binning: it creates ordinal types bins
pd.cut(x, bins = 3, labels = [1,2,3]) #equaly separated bins
pd.qcut(x, q = 3, labels = [1,2,3]) #equal number of data points in each bin

#pd.get_dummies(0) #obtain bolean that describe categories on a series
s = pd.Series(['A', 'B', 'A', 'C', 'D'])
boleanOfS = pd.get_dummies(s)
from sklearn.preprocessing import LabelEncoder
m = LabelEncoder()
m.fit(s)
LabelEncoder() 
m.transform(s)




