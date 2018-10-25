import requests
import re
import time

path = '../results/lyricsAnalysis'

def getSongs4Artists(listOfArtists, location = 'online'):
    if location == 'online':
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        songs4Artists = {}
        for artist in listOfArtists:
            artist = artist.lower()
            artist = re.sub(' ', '-', artist)
            url = 'http://www.metrolyrics.com/' + artist + '-lyrics.html' 
            print("Fetching " + artist + ' songs online ...')
            print(url)
            time.sleep(30)
            page = requests.get(url, headers=headers)
            html = page.text
        
            # Get all musics for each artist
            pattern = '''metrolyrics\.com\/(\w.+)-lyrics-''' + artist + '''\.html'''
            songs = re.findall(pattern, html)
            songs4Artists[artist] = songs
            #print(songs4Artists)
        
            # Print all songs for each artist
            songNames = []
            for i,j in enumerate(songs): 
                songNames.append(re.sub("-", " ", songs[i]))
            #print(artist + ' songs: '+ str(songNames))
        print('Done!')

    elif location == 'local':
        print('TODO: create songs4artists from local database')
    
    return songs4Artists
    
def getLyrics4Artists(songs4Artists, numSongs = 5, location = 'online'):
    if location == 'online':
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
        lyrics4Artists = {}
        for artist in songs4Artists:
            print("Fetching " + str(numSongs) + ' lyrics for ' + artist + ' ...')
            listLyrics = [] 
            for i,songName in enumerate(songs4Artists[artist]):
                if i<numSongs:
                    url = 'http://www.metrolyrics.com/' + songName + '-lyrics-' + artist +'.html'
                    #print(url)
                    time.sleep(30)
                    page = requests.get(url, headers=headers)
                    htmlSong = page.text
                    lyrics = re.findall('''<p class='verse'>(\D+)<\/p>''', htmlSong)
                    lyrics = " ".join(lyrics) # concatenates all verses in one string
                    lyrics = re.sub(r'<br>\\n', " ", lyrics)
                    lyrics = re.sub(r'<br>', " ", lyrics)
                    lyrics = re.sub(r'\"', "", lyrics)
                    lyrics = re.sub(r'<\/p><p class=\'verse\'>', " ", lyrics)
                    listLyrics.append(lyrics)
                
                    # Save song lyrics in a file
                    filename = re.sub('-', '', artist) + '_' + re.sub('-', '', songName) + '.txt'
                    f = open(path + '/' + filename, 'w')
                    f.write(lyrics)
                    f.close()
                    f = open(path + '/' + filename, 'r')
                    text = f.read()

            lyrics4Artists[artist] = listLyrics
        #print(lyrics4Artists)
        print('Done!')
    
    elif location == 'local':
        print('TODO: create lyrics4Artist from local database')
    
    return lyrics4Artists

def buildNaiveBayesModel(lyrics4Artists):
    labels = []
    lyrics = []
    for artist in lyrics4Artists.keys():
        for song in lyrics4Artists[artist]:
            labels.append(artist)
            lyrics.append(song)

    # tokenize + count bag of words
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    cv = CountVectorizer()
    vec = cv.fit_transform(lyrics)
    # apply Tf-Idf
    tf = TfidfTransformer()
    X = tf.fit_transform(vec) # normalise the vec data

    # build Naive Bayes model
    from sklearn.naive_bayes import MultinomialNB
    m = MultinomialNB()
    m.fit(X,labels)
    m.score(X,labels)
    return m, cv, tf

def proba_Lyrics4Artists(test_songs, m, cv, tf):

    # vectorize + tfidf first
    test_vec = cv.transform(test_songs)
    test_vec2 = tf.transform(test_vec)

    # prediction results
    prediction = m.predict(test_vec2)
    print('\n Test songs might belong to:')
    print(prediction)

    # class probabilities
    classProb = m.predict_proba(test_vec2)
    print('\n Each song probability from being from each artist:')
    print(classProb)

    logProb = m.feature_log_prob_
    print(logProb)
    return prediction, classProb, logProb

#def concatenate_list_data(listSongs):
#    result= ''
#    for element in listSongs:
#        result += str(element)
#    return result


