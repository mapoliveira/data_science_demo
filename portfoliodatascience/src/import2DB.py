def createSQLiteDB(dataDir, file1, file2, sep='\t'):
    '''
    Imports and creates a sqlite3 database from 2 files that have the same identifiers in the first collumn.
    Variable names will be assigned based on the file headings.
    '''
    import pandas as pd
    import sqlite3
    import sys
    sys.path.insert(0, '../drivers')

    df1 = pd.read_csv(dataDir + file1, sep='\t')    
    df2 = pd.read_csv(daraDir + file2, sep='\t')     
    
    df1.set_index(df1.iloc[0], inplace = True)
    df2.set_index(df2.iloc[0], inplace = True)
    print(df1.head())
    print(df2.head())

    df3 = pd.merge(df1, df2, left_index = True, right_index = True, how = 'outer')
    print(df3.head())


    DB_SETUP = """
    CREATE TABLE IF NOT EXISTS moviesDB (
        id VARCHAR(255),
        ordering INTEGER,
        title VARCHAR(255),
        language VARCHAR(255),
        isOriginalTitle BOOLEAN,
        averageRating FLOAT,
        numVotes INTEGER
    );
    """

    db = sqlite3.connect('movies.db')
    db.executescript(DB_SETUP)

    for i, row in df3.iterrows():
        query = 'INSERT INTO moviesDB VALUES (?,?,?,?,?,?,?)'
        db.execute(query, (i, row['ordering'], row["title"], row["language"], row["isOriginalTitle"], row["averageRating"], row["numVotes"]))

    db.commit()
    db.close()

    print('\n### Congrats! Database movies.db created! ###')

file1 = '../rawData/title.akas.tsv'
file2 = '../rawData/title.ratings.tsv'

createSQLiteDB(dataDir, file1, file2, sep='\t')


