import pandas as pd
import matplotlib.pyplot as plt
import glob2
import os
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def analyseNames(filenames):
    path = os.getcwd() + "/namesInUSA"
    allfiles = glob2.glob(path + filenames)
    frame = pd.DataFrame()
    list_ = []
    birthPerYear = []
    for file_ in allfiles:
     df = pd.read_csv(file_, delimiter=',', header=None )
     year = int(re.findall('\d+', file_) [0])
     df['year'] = year 
     list_.append(df)
     
    # Concatenate all data into one DataFrame
    frame = pd.concat(list_)
    frame.columns = ['name', 'gender', 'numBirth','year']
    
    def initial(s):
        return(s[0])
    frame['initial'] = frame['name'].apply(initial)
    
    totalNumberBirth = frame['numBirth'].sum()
    frame['perc'] = frame['numBirth']/totalNumberBirth*100
    print(totalNumberBirth)

    # Top 10 initials:
    print("\nTop initials:") 
    topInitials = frame.groupby('initial')['numBirth', 'perc'].sum()
    topInitials = topInitials.sort_values(by=['numBirth'], ascending=False)
    print(topInitials.head(10))

    # Top 10 birth years:
    print("\nTop birth years:")
    birthPerYear = frame.groupby('year')['numBirth', 'perc'].sum()
    birthPerYear = birthPerYear.sort_values(by=['numBirth'], ascending=False)
    print(birthPerYear.head(10))

    # Top 10 names:
    print("\nTop names:")
    birthPerName = frame.groupby('name')['numBirth', 'perc'].sum()
    birthPerName = birthPerName.sort_values(by=['numBirth'], ascending=False)
    topNames = birthPerName.head(10)
    print(topNames)
     
    ax = topNames[['numBirth']].plot(kind='bar')
    plt.show()
    plt.savefig("topInitials.png", bbox_inches='tight')   
    
    # Calculate number of births 
    #print('Number of births: ' + str(totalNumBirths))
    #top10 = frame.sort_values(by=['numBirth'], ascending=False).head(10)
    #top5 = top10.head(5).copy()
    #top5['perc'] = top5['numBirth']/(totalNumBirths)* 100
    #print(top5)

    #top5.set_index("name",drop=True,inplace=True) 
    #plot = top5.plot.bar()
    #fig = plot.get_figure()
    #fig.savefig("top5.png", bbox_inches='tight')
    
    # Machine learning for one name across the years ()
    X = frame[['year']].values
    y = frame['numBirth'].values
    # Linear model
    m = LinearRegression()
    m.fit(X,y)
    print(m.coef_)
    print(m.intercept_)
    ypred = m.predict(X)
    fig = plt.figure()
    plt.plot(X, y, 'bo')
    plt.plot(X, ypred, 'rx')
    plt.show()
    fig.savefig("linearModel.png")
   
    #poly = PolynomialFeatures(degree=2)
    #X_ = poly.fit_transform(X)
    #predict_ = poly.fit_transform(y)
    
   #--1986--#
#print('----Baby name analysis in 1986----')
#analyseNames("/yob1986.txt") 

#--All years--#- 
print('----Baby name analysis across all years----')
analyseNames("/*.txt")
