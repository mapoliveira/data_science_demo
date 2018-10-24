import requests
import re
import time

path = '../results/lyricsAnalysis'

def getSongs4Artists(listOfArtists):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    songs4Artists = {}
    for artist in listOfArtists:
        artist = artist.lower()
        artist = re.sub(' ', '-', artist)
        url = 'http://www.metrolyrics.com/' + artist + '-lyrics.html'
        print("\n" + url)
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
        print(artist + ' songs: '+ str(songNames))
    return songs4Artists
    
def getLyrics4Artists(songs4Artists):
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    lyrics4Artists = {}
    for artist in songs4Artists:
        listLyrics = [] 
        concatLyrics4Artist = []
        allWords4Artist = []
        for i,songName in enumerate(songs4Artists[artist]):
            if i<2:
                url = 'http://www.metrolyrics.com/' + songName + '-lyrics-' + artist +'.html'
                print(url)
                time.sleep(30)
                page = requests.get(url, headers=headers)
                htmlSong = page.text
                lyrics = re.findall('''<p class='verse'>(\D+)<\/p>''', htmlSong)
                lyrics = re.sub(r'<br>\\n', " ", str(lyrics))
                lyrics = re.sub(r'<\/p><p class=\'verse\'>', " ", str(lyrics))
                listLyrics.append(lyrics)
                
                # Save song lyrics in a file
                filename = re.sub('-', '', artist) + '_' + re.sub('-', '', songName) + '.txt'
                f = open(path + filename, 'w')
                f.write(lyrics)
                f.close()
                f = open(path + filename, 'r')
                text = f.read()

        concatLyrics4Artist.append(concatenate_list_data(listLyrics))
        lyrics4Artists[artist] = concatLyrics4Artist
    return lyrics4Artists

def buildNaiveBayesModel(lyrics4Artists):
    labels = list(lyrics4Artists.keys())
    
    # tokenize + count bag of words
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    cv = CountVectorizer()
    vec = cv.fit_transform(lyrics4Artists)
    # apply Tf-Idf
    tf = TfidfTransformer()
    X = tf.fit_transform(vec) # normalise the vec data
    
    # build Naive Bayes model
    from sklearn.naive_bayes import MultinomialNB
    m = MultinomialNB()
    m.fit(X,labels)
    m.score(X,labels)
    return m

def concatenate_list_data(listSongs):
    result= ''
    for element in listSongs:
        result += str(element)
    return result


