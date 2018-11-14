# Add movies to Django database from a sqlite3 database and add the movie poster

import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "movieRecommender.settings")

import django
django.setup()

from recommender.models import Movie
#for m in Movie.objects.all():
#    print(m)

"""
Options for writing existing data to a DB managed by Django:

1) read data files to a df, and loop that creates Django Model objects
2) read data from existing DB to a df (with pd.read_sql), and loop
3) read data from existing DB using sqlite3 and loop
4) .. write entries to Django DB with sqlite3 
     (requires to find out table + column names)
5) use SQL COPY to load CSV file to DB directly
     (maybe doesn't work with sqlite3)

"""
# Add all movies from an existing sqlite3 DB:
import pandas as pd
import sqlite3
import requests
import os

dbGroupLens = sqlite3.connect("moviesGroupLens.db")

def qGroupLens(query):
    # Query GroupLens database
    return pd.read_sql(query,dbGroupLens)

def getMovieCover(MOVIE_NAME):
    # Obtain movie cover for each movie
    BASE_URL = "http://www.omdbapi.com"
    API_KEY = "e20b83a2"

    params = {'t': MOVIE_NAME[:-6].strip(), 'apikey': API_KEY}
    r = requests.get(BASE_URL, params=params)
    j = r.json()
    poster_url = j['Poster']
    return poster_url

df = qGroupLens("SELECT movieId, imdbId, userId, title, genres, rating FROM moviesGroupLens WHERE userID !='NaN' ORDER BY userId ASC")

## Obtain covers for movies
posters = [None] * len(df)

for index, MOVIE_NAME in enumerate(df['title']):
    if index<10:
        try:
            poster_url = getMovieCover(MOVIE_NAME)
            posters[index] = poster_url 
        except:
            posters[index] = '' 
            print('No poster found for: ' + MOVIE_NAME)
df['url'] = posters

print(df.head(10))


## Add movie to Django database:
for index, row in df.iterrows():
    m = Movie()
    m.userId = row['userId']
    m.rating = row['rating']
    m.imdbId = row['imdbId']
    m.movieId = row['movieId'] 
    m.title = row['title']
    m.genre = row['genres']
    #m.poster = row['url']
    m.save()

## Add a new movie by input:
#m = Movie()
#m.movie_id = input('enter a movie ID: ')
#m.title = input('enter a title: ')
#m.genre = input('enter a genre: ')
#m.save()


