#imports

import os               #for ENVIRON
import numpy as np      #for numpy arrays
import pandas as pd     #for dataframes
import spotipy          #for spotify search
import pickle           #for pickling our pipeline
import warnings         #for ignore
import re

from sqlalchemy import create_engine, URL, text     #for database access
from sklearn.cluster import KMeans                  #for KMeans algorithm
from sklearn.preprocessing import StandardScaler    #for scaling data
from sklearn.pipeline import Pipeline               #for ML pipelines
from scipy.spatial.distance import cdist            #for calculating distances of songs
from collections import defaultdict                 #for dicts
from spotipy.oauth2 import SpotifyClientCredentials #for spotify search
from flask import Flask, jsonify, request, abort    #for flask application
from datetime import datetime                       #for data rebuilding

warnings.filterwarnings("ignore")

#varibles to tweak the program
numclusters = 10                    #how many clusters in KMeans
numsongs = 15                       #how many songs to output
saveupdate = False

#get environment variables

DB_SERVER = os.environ.get('DB_SERVER')
DB_PORT = os.environ.get('DB_PORT')
DB_USER = os.environ.get('DB_USER')
DB_PASS = os.environ.get('DB_PASS')
DB_TABLE = os.environ.get('DB_TABLE')

SPOTIPY_CLIENT_ID = os.environ.get('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.environ.get('SPOTIPY_CLIENT_SECRET')

#create connection URL

conn_url = URL.create("mssql+pyodbc",
                      username=DB_USER,
                      password=DB_PASS,
                      host=DB_SERVER,
                      port=DB_PORT,
                      database=DB_TABLE,
                      query={
                          'driver': 'ODBC DRIVER 18 for SQL Server',
                          'TrustServerCertifcate': 'yes',
                          'authentication': 'ActiveDirectoryPassword'
                      })

#create SQLAlchemy Engine

engine = create_engine(conn_url)

#create SQL query string

sql = ( 'SELECT '
        'S.id, S.name, S.artists, SD.danceability, SD.energy, SD.loudness, SD.mode, SD.speechiness, '
        'SD.acousticness, SD.instrumentalness, SD.liveness, SD.valence, SD.tempo, SD.year '
        'FROM Songs as S INNER JOIN SongData as SD ON (S.id = SD.id)')

#read SQL from database into DataFrame && remove duplicate columns (to remove duplicate ID rows)

data = pd.read_sql(sql, engine)
data = data.loc[:, ~data.columns.duplicated()]

#establish connection to spotipy API

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID, 
    client_secret=SPOTIPY_CLIENT_SECRET))

#create pipeline for KMeans

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                  ('kmeans', KMeans(n_clusters=numclusters,
                                   verbose=False))
                                  ], verbose=False)

#create columns that data is pipeline is built with

number_col = ['danceability', 'energy', 'loudness', 'speechiness', 
              'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

#sends song to database when not present
def new_song(song):
    print('Entering New Song')

    #get values of id, name, album_id
    #create list of the lists of artists and artists_id
    id = song['id'].values[0]
    name = song['name'].values[0].replace('\'', '\'\'')
    album_id = song['album_id'].values[0].replace('\'', '\'\'')
    artists = list(song['artists'])[0]
    artists_id = list(song['artists_id'])[0]

    #for each artist format in format used in SQL database
    artists_formatted = '['
    for a in artists:
        artists_formatted = artists_formatted + a + ', '
    artists_formatted = artists_formatted[:-2] + ']'
    
    #for each artist_id format in format used in SQL database
    artists_id_formatted = '['
    for a in artists_id:
        artists_id_formatted = artists_id_formatted + a + ', '
    artists_id_formatted = artists_id_formatted[:-2] + ']'

    #create sql query to insert songs into song database
    insert_sql_song = ('INSERT INTO Songs '
                       '(id, name, album_id, artists, artists_id) '
                       'VALUES '
                       '(\'{}\', \'{}\', \'{}\', \'{}\', \'{}\')'.format(id, name, album_id, artists_formatted, artists_id_formatted))

    #create sql query to insert song data into database
    insert_sql_song_data = ('INSERT INTO SongData '
                            '(id, explicit, danceability, energy, [key], loudness, mode, speechiness, acousticness, '
                            'instrumentalness, liveness, valence, tempo, duration, time_signature, year) '
                            'VALUES '
                            '(\'{}\', {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})'.format(id, song['explicit'].values[0], song['danceability'].values[0], song['energy'].values[0], 
                                                                                                      song['key'].values[0], song['loudness'].values[0],song['mode'].values[0], song['speechiness'].values[0], 
                                                                                                      song['acousticness'].values[0], song['instrumentalness'].values[0], song['liveness'].values[0], song['valence'].values[0],
                                                                                                      song['tempo'].values[0], song['duration_ms'].values[0], song['time_signature'].values[0], song['year'].values[0]))
    
    #create sql query to check if song exists already
    select_sql_song = ('SELECT id FROM Songs WHERE id=\'' + id + '\'')
    
    #with connection, check if song is present in database
    #if not present, then insert song into database and commit
    with engine.connect() as conn:
        result = conn.execute(text(select_sql_song))
        if result.fetchone() is None:
            print("Entered Song Into Database")
            conn.execute(text(insert_sql_song))
            conn.execute(text(insert_sql_song_data))
            conn.commit()

    print('Exiting New Song\n')
    return 0

#fits the pipeline with data
def fit_pipeline(pipeline):
    print('Entering Fit Pipeline')
    #maybe not the best solution, but
    #fetch global variable which contains whether it updated
    #prevents wednesday recreating the pipeline over and over
    #but will ultimately cause it to not update twice unless
    #the program has been restarted
    global saveupdate

    #if pipeline data is 0 or if the weekday is wednesday - recreate data.sav
    #as we add data into our database we need to recreate data.sav to allow for the music to be recommended
    if ((os.path.getsize('data/data.sav') == 0 or datetime.today().weekday() == 2) and (not saveupdate)):
        #take the values of data w/o axes
        #fit pipeline with the data
        #assign labels to our data
        X = data[number_col].values
        pipeline.fit(X)
        song_cluster_labels = pipeline.predict(X)
        data['cluster_label'] = song_cluster_labels

        saveupdate = True
        #pickle the data.sav
        pickle.dump(pipeline, open('data/data.sav', 'wb'))
        print('Pipeline Created!')
    else:
        #unpickle the data.sav
        pipeline = pickle.load(open('data/data.sav', 'rb'))
        print('Pipeline Loaded!')

    print('Exiting Fit Pipeline\n')
    return pipeline

#finds the song in Spotify API
def find_song(name, artists, year=None):
    print('Entering Find Song')

    song_data = defaultdict()
    correctsong = None
    #search Spotify for song with same name, artist and release year
    if year is None:
        results = sp.search(q='track:{} artist:{}'.format(
                            name, artists), limit=50, market='US', type='track')
    else:
        results = sp.search(q='track:{} artist:{} year:{}'.format(
                            name, artists, year), limit=50, market='US', type='track')
    
    #if results is empty (means no song was found with that information)
    if results['tracks']['items'] == []:
        print('Results are Empty!')
        print('Exiting Find Song\n')
        return None
    
    #loop for each song in results (total of 50 can be saved) (probably can reduce this)
    for song in results['tracks']['items']:
        #check if inputted name is in song name)
        if (name.casefold() in song['name'].casefold()):
            #loop thru artists and check if any artist name matches the list
            for a in song['artists']:
                #if so select song as correct song
                if(a['name'].casefold() == artists.casefold()):
                    correctsong = song
                    break
            continue
        else:
            #name incorrect go to next song
            continue
    
    #if no correct song is found
    if (correctsong == None):
        print('No Correct Song')
        print('Exiting Find Song\n')
        return None

    #extract name, artist, year, id from the song
    track_name = correctsong['name']
    track_artist = correctsong['artists'][0]['name']
    track_year = correctsong['album']['release_date'][0:4]
    track_id = correctsong['id']
    album_id = correctsong['album']['id']
    track_explicit = (1 if correctsong['explicit'] else 0)

    track_artists = []
    artists_ids = []
    for artist in correctsong['artists']:
        track_artists.append(artist['name'])
        artists_ids.append(artist['id'])

    #search spotify with ID to get audio features (danceability, energy etc.)
    audio_features = sp.audio_features(track_id)[0]

    #assign infomation to a new variable
    song_data['name'] = [track_name]
    song_data['year'] = [int(track_year)]
    song_data['artists'] = [track_artists]
    song_data['artists_id'] = [artists_ids]
    song_data['album_id'] = [album_id]
    song_data['explicit'] = [track_explicit]

    #for all audio features assign key and value
    for key, value in audio_features.items():
        song_data[key] = value

    print('Created Data Frame for Found Song')
    print('Exiting Find Song\n')
    return pd.DataFrame(song_data)

#retrieves the song data from Database 
def get_song_data(song):
    print('Entering Get Song Data')

    #try and except
    try:
        #search data for song that matches with song's ID
        song_data = data[ data['id'] == song['id'] ].iloc[0]
        print('Got Song Data from Database')
        print('Exiting Get Song Data\n')
        return song_data

    except IndexError:
        #if end of data is reached, search Spotify for the song!
        foundsong = find_song(song['name'], song['artists'], song['year'])
        #No song found
        if (foundsong) is None:
            print('Song was not Found on Spotify')
            print('Exiting Get Song Data\n')
            return None
        #add new song to database
        else:
            new_song(foundsong)

        print('Found Song on Spotify')
        print('Exiting Get Song Data\n')
        return foundsong

#calculates the mean vector for the songs inputed
def get_mean_vector(song_list):
    print('Entering Get Mean Vector')

    song_vectors = []

    #for each song in songs (adds ability to add multiple songs)
        #multiple songs is not really a good idea as it finds the middle between all the songs
        #this recommends song around that point and will probably not recommend songs related to the
        #songs inputted, unless songs are close together
    for song in song_list:
        #get the song data
        song_data = get_song_data(song)
        #no song found
        if song_data is None:
            print('Warning: "{}" does not exist in Spotify or in database'.format(
                song['name']))
            print('Exiting Get Mean Vector\n')
            return None
        
        #otherwise assign the .vaules of the data
        song_vector = song_data[number_col].values
        song_vectors.append(song_vector)

    #create a np.array of lists of the song vectors
    song_matrix = np.array(list(song_vectors))
    print('Exiting Get Mean Vector\n')
    #calculate the mean of the matrix
    return np.mean(song_matrix, axis=0)

#flattens dictionary 
def flatten_dict_list(dict_list):
    print('Entering Flatten Dict List')

    flattened_dict = defaultdict()
    #for each key in dictonary, create that key in the flat dict
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    #for each dictionary in dict_list, assign the key and values
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)

    print('Exiting Flatten Dict List\n')
    return flattened_dict

#reccomends songs based on song inputs
def recommend_songs(song_list, n_songs=numsongs):
    print('Entering Recommend Songs')

    metadata_cols = ['name', 'year', 'artists', 'id', 'album art']

    #assign the fitted pipeline
    sc_pipeline = fit_pipeline(song_cluster_pipeline)

    #get the flattened dict of songs
    song_dict = flatten_dict_list(song_list)

    #get the mean vectors of the songs (matrix)
    song_center = get_mean_vector(song_list)
    #if song center is None, means that song was not able to be located
    if (song_center is None):
        print('No Song Located in Spotify or Database')
        print('Exiting Recommend Songs\n')
        return 404
    #select the standard scaler() 
    scaler = sc_pipeline.steps[0][1]
    #scale the data using standard scaler()
    scaled_data = scaler.transform(data[number_col])
    #reshape data into 1, -1 (where -1 conforms to data)
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    #calculate the dist from the two datas
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    #create an index of the songs with the shortest distances
    index = list(np.argsort(distances)[:, :n_songs][0])

    #assing rec_songs the songs relating to the indexs
    rec_songs = data.iloc[index]
    #remove any song that contains the same name and artists as song_dict
    rec_songs = rec_songs[~(rec_songs['name'].isin(song_dict['name']))]
    #collect extra data for songs (album art)
    rec_songs = extra_data(rec_songs)
    print('Exiting Recommend Songs\n')
    #create a dictionary of the songs with records for {columns => values}
    return rec_songs[metadata_cols].to_dict(orient='records')

#retrieves extra song data (album art)
def extra_data(song_data):
    print('Entering Extra Data')

    #retrieve song with a reset index
    ids = song_data['id'].reset_index().to_dict()
    art = []
    #for each id, get the spotify search of the data and assign album art
    for id in ids['id'].values():
        song = sp.track(id)
        art.append(song['album']['images'][0]['url'])
    
    #assign art sources to song_data
    song_data['album art'] = art
    print('Exiting Extra Data\n')
    return song_data

# -- Flask Program --

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_home():
    #empty home page 
    return jsonify('Hello!')


@app.route('/api/song/', methods=['GET'])
def get_song():
    #api song testing pre songs
    test = [{'name': 'Testify', 'artists': "Rage Against The Machine", 'year': 1999, 'id': '7lmeHLHBe4nmXzuXc0HDjk'}]
    test2 = [{'name': 'How to Save A Life', 'artists': 'The Fray', 'year': 2005, 'id': ''}, 
            {'name': 'If I Die Young', 'artists': 'The Band Perry', 'year': 2010, 'id': ''}, 
            {'name': 'Somebody That I Used To Know', 'artists': 'Gotye', 'year': 2011, 'id': ''}]
    return jsonify(recommend_songs(test))


@app.route('/api/song/id/<string:song_id>', methods=['GET'])
def get_recommended_id(song_id):
    #get song recommendation for song ID's
    #get song from data
    df = (data[data['id'] == song_id])
    #if empty (song doesn't exist in database)
    if (df.empty):
        return ("DataFrame is empty (song ID not found)")
    #get song info as compile a list of a dictionary
    else:
        name = df['name'].to_list()
        artist = df['artists'].to_list()
        year = df['year'].to_list()
        song_input = [{'name': ', '.join(name), 'artists': ', '.join(
            artist), 'year': int(', '.join(map(str, year))), 'id': song_id}]

    result = recommend_songs(song_input)

    if result == 404:
        abort(404)

    return jsonify(result)

    """default = [{'name': 'It Must Be a Pain',
                'artists': 'sewerperson', 'year': 2022}]
    return jsonify(recommend_songs(default, data))
    # return jsonify(recommend_songs(song_id, data))"""


@app.route('/api/song/search', methods=['GET'])
def get_recommended_search():
    #search song by name and artists
    #format for search is ../api/song/search?track=<TRACKINFO>&artist=<ARTISTINFO>
    #get both track name and artist
    song_name = request.args.get('track')
    artist = request.args.get('artist')

    #make sure both artist and name is present
    if artist is None: 
        return jsonify('No artist found!')
    if song_name is None:
        return jsonify('No song name found!')
    
    #find song in Spotify
    song = find_song(song_name, artist)

    #if none (song not found)
    if song is None:
        abort(404)
    #assign info to list of a dictionary
    else:
        name = song['name'].values
        artist = list(song['artists'])[0]
        year = song['year'].values

        if (artist[:-len(artist)+1] == []):
            artist = artist
        else:
            artist = artist[:-len(artist)+1]
        
        song_input = [{'name': ', '.join(name), 'artists': ', '.join(artist), 
                    'year': int(', '.join(map(str, year))), 'id': None}]
    
    print(song_input)

    result = recommend_songs(song_input)

    if result == 404:
        abort(404)

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
