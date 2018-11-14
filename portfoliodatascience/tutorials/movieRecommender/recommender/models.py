''' This defines the class Movie which refere to the database entries.'''

from django.db import models

class Movie(models.Model):
    movieId = models.IntegerField()
    title = models.TextField()
    genre = models.TextField()
    userId = models.IntegerField()
    rating = models.IntegerField()
    imdbId = models.IntegerField()
    #posterUrl = models.TextField() 

def __str__(self):
    return f"{self.movie_id} : {self.title}"
