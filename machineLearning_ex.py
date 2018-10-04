#### MACHINE LEARNING EXERCISE ####
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random 
import numpy as np

x = list(range(20))
y = [3* xx + 10 *random.random() for xx in x]  # list of linear dependent values with noise
plt.show()
# scikit learn need a specific shape for the data
X = np.array(x).reshape(-1, 1)
Y = np.array(y)
print(X.shape, y.shape)

m = LinearRegression()
m.fit(X,y)

print(m.coef_)
print(m.intercept_)
ypred = m.predict(X)

plt.figure()
plt.plot(X, y , 'bo')
plt.plot(X, ypred, 'rx')
plt.show()
