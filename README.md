# Python Machine Learning Music Recommendation System
A Flask application that will allow for input of an ID or, Song name and Artist

## DEPRECATED AS OF November 27, 2024

### Due to Spotify's API changes, the functionality of the recommendation system is no longer functional as per implemented here 

Used in part of [**Mik's Music**](https://github.com/itsnotmik/nodejs-miks-music)

Will output a JSON format of the recommendations!

License: Apache 2.0

Expected changes:
   1. (completed) allow for read of data from a database instead of csv
   2. (completed) allow for writing of data to database
   3. (completed) allow for data.sav to regenerate (Need to modify this so that data.sav does not regen everytime on wednesday but only once)
   4. (completed) set up code to not recommend songs of the same name (currently recommending songs if case sensitive title is incorrect)
   5. allow for spotipy to include recommendations as well
   6. (completed) create detailed documentation around code
   7. (working) include more code with increased functionality


Known Errors:
   1. System can only regenerate data.sav if system has restarted. (doesn't pose a problem to the code now, but in a final form, would be a problem)

Original system developed by: Vatsal Mavani, [Posted Here](https://www.kaggle.com/code/vatsalmavani/music-recommendation-system-using-spotify-dataset)

## If you use my code, I would appreciate a link to my github project!