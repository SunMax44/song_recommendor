#imports of libraries of both webscraping billboards hot100 and spotify api
from bs4 import BeautifulSoup
import pandas as pd
import requests
import numpy as np
import json
import spotipy
import pickle
from IPython.display import IFrame
from IPython.display import display
import random
import sys

#Initialize SpotiPy with user credentials
from config import client_id, client_secret
from spotipy.oauth2 import SpotifyClientCredentials

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id,
                                                           client_secret=client_secret))

bb_df = pd.read_csv('bbhot100.csv')

# def play song function
def play_song(track_id):
    return IFrame(src="https://open.spotify.com/embed/track/"+track_id,
       width="320",
       height="80",
       frameborder="0",
       allowtransparency="true",
       allow="encrypted-media",
      )

# get user input and check if i billboard hot 100
# if yes give random other hot 100 song
# if no give kmeans model recommendation based on audiofeatures

# Streamlit setup
# title
st.title("Spotify Song Recommendor")
# Text Input
text_input = st.text_input("Enter a song you like to get a recommendation for a song you might also like!")
st.write("Text Input:", text_input)

if st.button('Lets go!'):
    # bring user input in specific form
    song_artist_input = [item.strip().strip('"').lower() for item in text_input.split(',')]

    spotify_search = sp.search(q=song_artist_input, type='track', limit=10)
    for song in range(10):
        input_id = spotify_search['tracks']['items'][song]['id']
        spotify_player(input_id)
        input_song_name = spotify_search['tracks']['items'][song]['name']
        st.write(input_song_name)
        if st.text_input('Did you mean this song? Type in y for yes or n for no.') in ['y','yes']:
            break
        else:
            continue

    if input_song_name in bb_df['title'].values:
        # Get random row and column indices
        random_row = np.random.randint(0, bb_df.shape[0])
        random_title = bb_df.iloc[random_row, 0]
        random_artist = bb_df.iloc[random_row, 1]
        spotify_search2 = sp.search(q=random_title + ' ' + random_artist, limit=1)
        hot_rec_id = spotify_search2['tracks']['items'][0]['id']
        spotify_player(hot_rec_id)
        st.write(f'Your song is currently sizzling hot, so here is another hot song: {random_title} by {random_artist}')
        sys.exit()

    else:
        print("Your song is not in the Billboard Top 100, but of course this doesnt mean anything. Here is a recommendation for you:")

    # get audio features of the input song
    af=pd.DataFrame(sp.audio_features(input_id))
    af_relevant=af[["danceability","energy","loudness","speechiness","acousticness",
        "instrumentalness","liveness","valence","tempo"]]

    # Load the scaler and KMeans model
    with open('scaler.pkl', 'rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)

    with open('kmeans_8_cluster.pkl', 'rb') as kmeans_file:
        loaded_kmeans = pickle.load(kmeans_file)

    af_scaled = loaded_scaler.transform(af_relevant)

    cluster_predicted = loaded_kmeans.predict(af_scaled)

    afeatures_df = pd.read_csv('afeatures_with_clusters.csv')

    cluster_filtered = afeatures_df[afeatures_df['cluster'] == int(cluster_predicted)]

    random_row = np.random.randint(0, cluster_filtered.shape[0])
    recommendation_id = cluster_filtered.iloc[random_row, 12]

    spotify_player(recommendation_id)