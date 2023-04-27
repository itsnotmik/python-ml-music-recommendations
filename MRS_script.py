import os
import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'spotipy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pickle-mixin'])

import numpy as np
import pandas as pd
import spotipy
import pickle
import warnings

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist
from collections import defaultdict
from spotipy.oauth2 import SpotifyClientCredentials

warnings.filterwarnings("ignore")

data = pd.read_csv('E:\GitHub\music-recommendation-system\data\data_features.csv')

with open('E:\GitHub\music-recommendation-system\data\spotipyclientid.txt', 'r') as tf:
    SPOTIPY_CLIENT_ID = tf.read()
with open('E:\GitHub\music-recommendation-system\data\spotipyclientsecret.txt', 'r') as tf:
    SPOTIPY_CLIENT_SECRET = tf.read()
    
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), 
                                  ('kmeans', KMeans(n_clusters=10, 
                                   verbose=False))
                                 ], verbose=False)

number_cols = ['track_number', 'disc_number', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
               'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature', 'year']

def main():
    argutest = ' '.join(sys.argv[1:])
    print(argutest)
    argu = np.array(list(eval(' '.join(sys.argv[1:]))))
    print(argu)

    print(argu[1]['name'])

    print(recommend_songs(argu, data))
    return 0

def new_song(song):
    #sendsongtodatabase!
    return 0

def fit_pipeline(pipeline):
    if (os.path.getsize('E:\GitHub\music-recommendation-system\data\data.sav') != 0):
        X = data.select_dtypes(np.number)
        number_cols = list(X.columns)
        pipeline.fit(X)
        song_cluster_labels = pipeline.predict(X)
        data['cluster_label'] = song_cluster_labels

        pickle.dump(pipeline, open('E:\GitHub\music-recommendation-system\data\data.sav', 'wb'))
    else:
        pipeline = pickle.load(open('E:\GitHub\music-recommendation-system\data\data.sav', 'rb'))

    return pipeline

def find_song(name, artists, year):
    song_data = defaultdict()
    results = sp.search(q='track: {} artists: {} year: {}'.format(name,artists,year), limit=1, type='track')
    if results['tracks']['items'] == []:
        return None

    results = results['tracks']['items'][0]
    
    out_name = results['name']
    out_artist = results['artists'][0]['name']
    out_year = results['album']['release_date'][0:4]
    
    if (out_name.casefold() != name.casefold()) or (out_artist.casefold() != artists.casefold()):
        return None
    
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [out_name]
    song_data['year'] = [int(out_year)]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]
    song_data['track_number'] = [results['track_number']]
    song_data['disc_number'] = [results['disc_number']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

def get_song_data(song, spotify_data):
    
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) & (spotify_data['artists'] == song['artists']) & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    
    except IndexError:
        foundsong = find_song(song['name'], song['artists'], song['year'])
        if (foundsong) is None:
            return None
        else:
            new_song(foundsong)
        return foundsong
        

def get_mean_vector(song_list, spotify_data):
    
    song_vectors = []
    
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print('Warning: "{}" does not exist in Spotify or in database'.format(song['name']))
            return None
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)  
    
    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)


def flatten_dict_list(dict_list):
    
    flattened_dict = defaultdict()
    for key in dict_list[0].keys():
        flattened_dict[key] = []
    
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
            
    return flattened_dict


def recommend_songs(song_list, spotify_data, n_songs=10):

    sc_pipeline = fit_pipeline(song_cluster_pipeline)
    
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    
    song_center = get_mean_vector(song_list, spotify_data)
    if (song_center is None):
        return "Song Could Not Be Located in Spotify or Tunit Database"
    scaler = sc_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

if __name__ == '__main__':
    main()