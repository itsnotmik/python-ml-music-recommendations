# music-recommendation-system
 A Python Based Music Recommondation System

 Original work: Vatsal Mavani, https://www.kaggle.com/code/vatsalmavani/music-recommendation-system-using-spotify-dataset

 Modified work: Michael Clement

 Data used:

 License: Apache 2.0

 Expected changes:
    1.allow for read of data from a database instead of csv
    2.allow for writing of data (adding new songs to the data_files)
    3.possibly add new songs to the dict
    4.set up code to not recommend songs of the same name (currently recommending songs if case sensitive title is incorrect)
    5.allow for spotipy to include recommendations as well
    6.create detailed documentation around code
    7.include more code with increased functionality


Known Errors:
   1.(possibly fixed)Songs can no longer be recommendended unless they are spelled correctly (including caps)
      previously, it would recommend the same song inputed but could recommend song even if not capitalized correctly
   2.When searching spotify for a song/artist/year, it will usually grab the something with the correct information (even if all other info is wrong)
      ex. if you enter an artist, but enter a completely random song name (which the artist does not have), it will grab a random song of their's
                     (possible fix) read the spotify information and verify that 'name' and 'artists' is correct
