import requests
import re
import time
import numpy as np

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

def tokenLyrics4Artists(lyrics4Artists, method='countVectorize'):
    #print(lyrics4Artists)
    labels = []
    lyrics = []
    for artist in lyrics4Artists.keys():
        for song in lyrics4Artists[artist]:
            labels.append(artist)
            lyrics.append(song)
            
    if method == 'spacy':
        import spacy
        try:
            # Load English tokenizer, tagger, parser, NER and word vectors
            nlp = spacy.load('en_core_web_sm')
        except:
            import os
            os.system("python -m spacy download en_core_web_lg")
            nlp = spacy.load('en_core_web_sm')

        doc = nlp(lyrics)
    
    elif method == 'countVectorize':
        # tokenize + count words + Tfid normalization
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        cv = CountVectorizer(stop_words='english', min_df = 10 )
        #print(lyrics)
        vec = cv.fit_transform(lyrics)
        #print(vec)
    
        tf = TfidfTransformer()
        X = tf.fit_transform(vec) # normalise the vec data
        return X, cv, vec, tf, labels, lyrics

def buildNaiveBayesModel(X, labels):
    from sklearn.naive_bayes import MultinomialNB
    m = MultinomialNB()
    m.fit(X, labels)
    m.score(X, labels)
    return m

    #### Latent Dirichlet Allocation

    #lda = LatentDirichletAllocation(n_components=3)
    #lda.fit(vec)
    #c = lda.components_
    #words = list(sorted(cv.vocabulary_.keys()))

    #ctrans = c.T

    #df = pd.DataFrame(ctrans, index=words)

    #for i in range(10):
    #    print(df[i].sort_values(ascending=False).head(20))

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

def wordcloud4Artist(lyrics4Artists, artist):
    allWords4Artist = " ".join(lyrics4Artists[artist])
    import wordcloud
    import matplotlib.pyplot as plt
    plt.figure(num = None, figsize = (20,20))
    
    wordcloud1 = wordcloud.WordCloud(background_color="white", max_words=2000, contour_color='steelblue').generate(allWords4Artist)
    plt.subplot(3,1,1)
    plt.title(artist, fontsize=18, loc='right')
    plt.imshow(wordcloud1, interpolation='bilinear')
    plt.show()

def cosine_similarity(a, b):
    """Returns cosine similarity of two word vectors"""
    return np.linalg.norm(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def analysis_gensimModels(artistLyrics, word1, word2, word3):
    from gensim.models import word2vec
    import numpy as np
    tokenized = [s.lower().split() for s in artistLyrics]
    wv = word2vec.Word2Vec(tokenized, size=7, window=5, min_count=1)
    #most similar word in the corpus for this word
    cs_12 = cosine_similarity(wv.wv[word1], wv.wv[word2])
    cs_13 = cosine_similarity(wv.wv[word1], wv.wv[word3])
    print('Distance between: ')
    print('love and right ' + str(cs_12))
    print('love and wrong ' + str(cs_13))
    print('\nDifference between right and wrong: ' + str(round((cs_12-cs_13)*100, 2)) + '%')
