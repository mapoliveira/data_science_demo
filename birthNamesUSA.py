import pandas as pd
import matplotlib.pyplot as plt
import glob2
import os
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Machine learning for one name across multiple years
def searchNameUsage(frame, name2search, gender, method):
    name = frame[(frame['name'] == name2search) & (frame['gender']==gender)]
    X = name[['year']].values
    y = name['percPerYear'].values
    if method =="linear":
       # Linear model
       m = LinearRegression()
       m.fit(X,y)
       print(m.coef_)
       print(m.intercept_)
       ypred = m.predict(X)
       fig = plt.figure()
       plt.plot(X, y, 'bo')
       plt.plot(X, ypred, 'rx')
       plt.title("Frequency of " + str(name2search) + " "+ str(gender))
       fig.savefig("results/linearModel_" + str(name2search) + "_"+ str(gender) +".png")
       plt.close()
    else:
       print('Do polynomial!')
       #poly = PolynomialFeatures(degree=2)
       #X_ = poly.fit_transform(X)
       #predict_ = poly.fit_transform(y)

    return
# Loading, wrangling and data analysis:
def analyseNames(filenames, top):
    path = os.getcwd() + "/rawData/namesInUSA"
    allfiles = glob2.glob(path + filenames)
    frame = pd.DataFrame()
    list_ = []
    birthPerYear = []
    for file_ in allfiles:
     df = pd.read_csv(file_, delimiter=',', header=None )
     year = int(re.findall('\d+', file_) [0]) 
     df['year'] = year 
     totalNumberBirthPerYear = df[2].sum()
     df['percPerYear'] = df[2]/totalNumberBirthPerYear*100
     list_.append(df)
     
    # Concatenate all data into one DataFrame
    frame = pd.concat(list_)
    frame.columns = ['name', 'gender', 'numBirth','year','percPerYear']
    
    def initial(s):
        return(s[0])
    frame['initial'] = frame['name'].apply(initial)
    
    totalNumberBirth = frame['numBirth'].sum()
    frame['perc'] = frame['numBirth']/totalNumberBirth*100
    print(totalNumberBirth)

    # Top initials:
    print("\nTop initials:") 
    topInitials = frame.groupby('initial')['numBirth', 'perc'].sum()
    topInitials = topInitials.sort_values(by=['numBirth'], ascending=False)
    print(topInitials.head(top))

    # Top birth years:
    print("\nTop birth years:")
    birthPerYear = frame.groupby('year')['numBirth', 'perc'].sum()
    birthPerYear = birthPerYear.sort_values(by=['numBirth'], ascending=False)
    print(birthPerYear.head(top))

    # Top names:
    print("\nTop names:")
    birthPerName = frame.groupby('name')['numBirth', 'perc'].sum()
    birthPerName = birthPerName.sort_values(by=['numBirth'], ascending=False)
    topNames = birthPerName.head(top)
    print(topNames)
     
    #ax = topNames[['numBirth']].plot(kind='bar')
    #plt.show()
    #plt.savefig("topInitials.png", bbox_inches='tight')   
    
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
    
    return(frame)

#------ 1986 ------#
print('1. Baby name analysis in 1986 ----------')
#analyseNames("/yob1986.txt", 10) 

#------ All years ------# 
print('2. Baby name analysis across all years ----------')
analyseNames("/*.txt", 10)

# Analyse names usage across all years:
searchNameUsage("Teresa", "M", "linear")
searchNameUsage("Teresa", "F", "linear")


