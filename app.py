import os
import numpy as np
import pandas as pd
import spotipy
import pickle
import warnings
import pyodbc

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist
from collections import defaultdict
from spotipy.oauth2 import SpotifyClientCredentials
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin


warnings.filterwarnings("ignore")

DB_SERVER = os.environ.get('DB_SERVER')
DB_USER = os.environ.get('DB_USER')
DB_PASS = os.environ.get('DB_PASS')
DB_TABLE = os.environ.get('DB_TABLE')

db_conn = pyodbc.connect("Driver={ODBC Driver 18 for SQL Server};"
                         "Server=" + DB_SERVER +
                         ";Database=" + DB_TABLE +
                         ";Uid=" + DB_USER +
                         ";Pwd=" + DB_PASS +
                         ";Encrypt=yes;"
                         "TrustServerCertificate=no;"
                         "Connection Timeout=30;"
                         "Authentication=ActiveDirectoryPassword")

sql = 'SELECT * FROM Songs as S INNER JOIN SongData as SD ON (S.id = SD.id)'

data = pd.read_sql(sql, db_conn)

SPOTIPY_CLIENT_ID = os.environ.get('SPOTIPY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.environ.get('SPOTIPY_CLIENT_SECRET')

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID, 
    client_secret=SPOTIPY_CLIENT_SECRET))

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                  ('kmeans', KMeans(n_clusters=10,
                                   verbose=False))
                                  ], verbose=False)

number_col = ['danceability', 'energy', 'key', 'loudness', 'speechiness', 
              'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']


def new_song(song):
    # sendsongtodatabase!
    return 0


def fit_pipeline(pipeline):
    print('Entering Fit Pipeline')

    if (os.path.getsize('data/data.sav') == 0):
        X = data[number_col]
        number_cols = list(X.columns)
        pipeline.fit(X)
        song_cluster_labels = pipeline.predict(X)
        data['cluster_label'] = song_cluster_labels

        pickle.dump(pipeline, open('data/data.sav', 'wb'))

        print('Pipeline Created!')
    else:
        pipeline = pickle.load(open('data/data.sav', 'rb'))
        print('Pipeline Loaded!')

    print('Exiting Fit Pipeline\n')
    return pipeline


def find_song(name, artists, year):
    print('Entering Find Song')
    song_data = defaultdict()
    results = sp.search(q='track: {} artists: {} year: {}'.format(
        name, artists, year), limit=50, type='track')
    if results['tracks']['items'] == []:
        print('Results are Empty!')
        print('Exiting Find Song\n')
        return None
    
    correctsong = None
    
    for song in results['tracks']['items']:
        if (song['name'].casefold() != name.casefold()) or (song['artists'][0]['name'].casefold() != artists.casefold()):
            continue
        else:
            correctsong = song
            break

    if (correctsong == None):
        print('No Correct Song')
        print('Exiting Find Song\n')
        return None

    out_name = correctsong['name']
    out_artist = correctsong['artists'][0]['name']
    out_year = correctsong['album']['release_date'][0:4]

    track_id = correctsong['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [out_name]
    song_data['year'] = [int(out_year)]
    song_data['explicit'] = [int(correctsong['explicit'])]
    song_data['duration_ms'] = [correctsong['duration_ms']]
    song_data['popularity'] = [correctsong['popularity']]
    song_data['track_number'] = [correctsong['track_number']]
    song_data['disc_number'] = [correctsong['disc_number']]

    for key, value in audio_features.items():
        song_data[key] = value

    print('Created Data Frame for Found Song')
    print('Exiting Find Song\n')
    return pd.DataFrame(song_data)


def get_song_data(song, spotify_data):
    print('Entering Get Song Data')

    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) & (
            spotify_data['artists'] == song['artists']) & (spotify_data['year'] == song['year'])].iloc[0]
        print('Got Song Data from Database')
        print('Exiting Get Song Data\n')
        return song_data

    except IndexError:
        foundsong = find_song(song['name'], song['artists'], song['year'])
        if (foundsong) is None:
            print('Song was not Found on Spotify')
            print('Exiting Get Song Data\n')
            return None
        else:
            new_song(foundsong)

        print('Found Song on Spotify')
        print('Exiting Get Song Data\n')
        return foundsong


def get_mean_vector(song_list, spotify_data):
    print('Entering Get Mean Vector')

    song_vectors = []

    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: "{}" does not exist in Spotify or in database'.format(
                song['name']))
            print('Exiting Get Mean Vector\n')
            return None
        
        song_vector = song_data[number_col].values
        song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    print('Exiting Get Mean Vector\n')
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    print('Entering Flatten Dict List')

    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []

    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)

    print('Exiting Flatten Dict List\n')
    return flattened_dict


def recommend_songs(song_list, spotify_data, n_songs=15):
    print('Entering Recommend Songs')

    sc_pipeline = fit_pipeline(song_cluster_pipeline)

    metadata_cols = ['name', 'year', 'artists', 'id', 'album art']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    if (song_center is None):
        print('No Song Located in Spotify or Database')
        print('Exiting Recommend Songs\n')
        return "Song Could Not Be Located in Spotify or Database"
    scaler = sc_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_col])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    rec_songs = extra_data(rec_songs)
    print('Exiting Recommend Songs\n')
    return rec_songs[metadata_cols].to_dict(orient='records')


def extra_data(song_data):
    print('Entering Extra Data')

    ids = song_data['id'].reset_index().to_dict()
    art = []
    for id in ids['id'].values():
        song = sp.track(id)
        art.append(song['album']['images'][0]['url'])
    
    song_data['album art'] = art
    print('Exiting Extra Data\n')
    return song_data

# Rest of code is to host application as server
# include...
# from flask import Flask, jsonify, request
# from flask_cors import CORS
# if not already inclueded


app = Flask(__name__)
#CORS(app)

@app.route('/', methods=['GET'])
#@cross_origin()
def get_home():
    return jsonify('Hello!')


@app.route('/api/song/', methods=['GET'])
#@cross_origin()
def get_song():
    test2 = [{'name': 'Testify', 'artists': "Rage Against The Machine", 'year': 1999}]
    test = [{'name': 'How to Save A Life', 'artists': 'The Fray', 'year': 2005}, 
            {'name': 'If I Die Young', 'artists': 'The Band Perry', 'year': 2010}, 
            {'name': 'Somebody That I Used To Know', 'artists': 'Gotye', 'year': 2011}]
    return jsonify(recommend_songs(test2, data))


@app.route('/api/song/<string:song_id>', methods=['GET'])
#@cross_origin()
def get_recommended(song_id):
    df = (data[data['id'] == song_id])
    if (df.empty):
        return ("DataFrame is empty (song ID not found)")
    else:
        name = df['name'].to_list()
        artist = df['artists'].to_list()
        year = df['year'].to_list()
        song_input = [{'name': ', '.join(name), 'artists': ', '.join(
            artist), 'year': int(', '.join(map(str, year)))}]

        return jsonify(recommend_songs(song_input, data))

    """default = [{'name': 'It Must Be a Pain',
                'artists': 'sewerperson', 'year': 2022}]
    return jsonify(recommend_songs(default, data))
    # return jsonify(recommend_songs(song_id, data))"""


if __name__ == '__main__':
    app.run(host='0.0.0.0')
