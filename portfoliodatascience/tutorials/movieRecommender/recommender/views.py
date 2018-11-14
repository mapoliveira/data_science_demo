''' This file defines the functions for visualization'''

from django.shortcuts import render
from .recommender import recommend_movies
from .models import Movie 

def main(request):
    movies = recommend_movies()
    all_movies = list(Movie.objects.all()) # SELECT * FROM movieRecommender GROUPBY movieId
    title = 'THE Movie Recommender'
    return render(request, 'hello.html',
            context={'movies': movies, 
                     'title': title,
                     'all': all_movies})

