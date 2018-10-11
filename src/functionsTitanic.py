
import pandas as pd

######################################### Feature engineering data:
def featureEngineering(df):
    print("Creating boleans categories for each variable...")
    listData = [] #List of data frames
    for i in df.columns:
        if i == 'Survived':
            y = df[i]
        elif i == 'Pclass' or i == 'SibSp' or i == 'Parch' or i == 'Embarked':
            dummies = pd.get_dummies(df[i])
            labels = list(range(1, dummies.shape[1] + 1))
            labels = [i + "_" + str(x) for x in labels]
            dummies.columns = labels
            listData.append(dummies)
        elif i == 'Name' or i == 'Ticket':
            print(i)
        elif i == 'Age' or i == 'Fare':
            bins = 4
            labels = list(range(1, bins + 1))
            labels = [i + "_" + str(x) for x in labels]
            dataBins = pd.cut(df[i], bins, labels)
            print(dataBins)
        elif i == 'Cabin':
            print(i)
        elif i == 'Sex':
            listData.append(pd.get_dummies(df[i]))
        else:
            print(i)

    newDf = pd.concat(listData, axis=1) # Concatenate data frames
    print(newDf.head(10))
    numCategories = newDf.shape[0]
    print("\nNumber of bolean categories based in all variables (except survival): "+ str(numCategories))
    return newDf

