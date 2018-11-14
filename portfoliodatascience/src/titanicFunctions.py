import pandas as pd
import re

### Feature engineering data for titanic dataset ###

def featureEngineering(df, features):
    print("Creating boleans categories for each variable...")
    listData = [] #List of data frames
    print(df.isna().any())
    #print(df.isna().any())
    #df.dropna(subset=['Age'], inplace = True) 
    #print(df.head(20))
    meanAge = df["Age"].mean()
    meanFare = df["Fare"].mean()
    #meanFare = df["Fare"].mean()
    #print(df['Embarked'].isna()==True)
    print("Non available ages were filled with average age: " + str(meanAge))
    #print("Non available fares were filled with average age: " + str(meanFare))
    df['Age'].fillna(meanAge, inplace=True)
    df['Fare'].fillna(meanFare, inplace=True)

    print(df.isna().any())

    for i in df.columns:
        if i == 'Survived':
            y = df[i]
        elif i == 'Pclass' or i == 'SibSp' or i == 'Parch' or i == 'Embarked':
            if i != 'Embarked':
                df[i]= df[i].apply(pd.to_numeric)
            dummies = pd.get_dummies(df[i])
            labels = list(range(1, dummies.shape[1] + 1))
            labels = [i + "_" + str(x) for x in labels]
            dummies.columns = labels
            print(dummies.isna().any())
            listData.append(dummies.iloc[:, :-1])
#        elif i == 'Name':
#            # Identify titles 
#            passengers = df[i]
#            title = []
#            name = []
#            index = df.index
#            for p in passengers:
#                t = re.search(', (.*)\.', p)
#                title.append(t.group(1))
#                
#                n = re.search('(.*),', p)
#                name.append(n.group(1))
#                
#            title = pd.Series(title, index = index) 
#            dummiesTitle = pd.get_dummies(title)
#            print(dummiesTitle.isna().any())
#            listData.append(dummiesTitle.iloc[:, :-1])
#            
#            name = pd.Series(name, index = index) #range(1, len(name)+1)) 
#            dummiesName = pd.get_dummies(name)
#            print(dummiesName.isna().any())
#            listData.append(dummiesName.iloc[:, :-1])

        elif i == 'Fare' or i == 'Age':
            df[i]= df[i].apply(pd.to_numeric)
            bins = 4
            labels = list(range(1, bins + 1))
            labels = [i + "_" + str(x) for x in labels]
            dataBins = pd.cut(df[i], bins, labels)
            dummies = pd.get_dummies(dataBins)
            print(dummies.isna().any())
            listData.append(dummies.iloc[:, :-1])
        elif i == 'Sex':
            dummies = pd.get_dummies(df[i])
            listData.append(dummies.iloc[:, :-1])
            print(dummies.isna().any())
        else:
            print(i + ' data was not considered')
    newDf = pd.concat(listData, axis=1) # Concatenate data frames
    #print(listData)
    #print(newDf)
    #print(newDf.isna().any())
    #assert (newDf.isna().any() == False), 'There are NaN values in the dummies!'
    
    # Identify best features
    if features == None:
        import numpy as np
        from sklearn.linear_model import Lasso
        #X = fit_transform(newDf) 
        print(newDf.columns)
        m = Lasso(alpha=0.0007).fit(newDf, y)
        print("Lasso coeficients:")    
        nonZeroLassoFeatures = np.where(m.coef_!=0)
        nonZeroLassoFeatures = np.array(nonZeroLassoFeatures).tolist()[0]
        print(nonZeroLassoFeatures)
        newDfLasso = newDf.iloc[:, nonZeroLassoFeatures]
        numFeatures = newDf.shape[1]
        selectFeatures = newDfLasso.shape[1]
        print("\nNumber of bolean categories based in all variables (except survival): "+ str(selectFeatures) + '/' + str(numFeatures))
        newDf = newDfLasso
        features = nonZeroLassoFeatures
    else:
        print(features)
        print(newDf.head(10))
        print(newDf.shape[1])
        print(newDf.columns)
        newDf = newDf.iloc[:, features]
        print(newDf.columns)
    return newDf, features

