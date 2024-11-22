#imports of libraries of both webscraping billboards hot100 and spotify api
from bs4 import BeautifulSoup
import pandas as pd
import requests
import numpy as np
import json
import spotipy
import pickle
import sys
import streamlit as st
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.preprocessing import StandardScaler  # for the imported scaler
from sklearn.cluster import KMeans # for the imported KMeans

# setup spotify API access
client_id = st.secrets["client_id"]
client_secret = st.secrets["client_secret"]
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id,
                                                           client_secret=client_secret))

# load billboard hot 100 dataframe
bb_df = pd.read_csv('bbhot100.csv')

import streamlit as st
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy

# Define Spotify player
def spotify_player(track_id):
    embed_url = f"https://open.spotify.com/embed/track/{track_id}"
    st.components.v1.iframe(embed_url, width=300, height=80)

# Streamlit setup
st.title("Spotify Song Recommender")
text_input = st.text_input("Enter a song you like to get a recommendation for a song you might also like!")

# Initialize session state to manage loop
if "current_song_index" not in st.session_state:
    st.session_state.current_song_index = 0
if "track_ids" not in st.session_state:
    st.session_state.track_ids = []
if "track_names" not in st.session_state:
    st.session_state.track_names = []

if st.button("Let's go!"):
    # Preprocess user input
    #song_artist_input = [item.strip().strip('"').lower() for item in text_input.split(',')]

    # Search for the song on Spotify
    spotify_search = sp.search(q=text_input, type='track', limit=10)

    # Store results in session state
    st.session_state.track_names = [song['name'] for song in spotify_search['tracks']['items']]
    st.session_state.track_ids = [song['id'] for song in spotify_search['tracks']['items']]
    st.session_state.current_song_index = 0  # Reset the index

# Display the current song in the loop
if st.session_state.track_ids:
    # Get the current song and its ID
    current_index = st.session_state.current_song_index
    current_song_name = st.session_state.track_names[current_index]
    current_song_id = st.session_state.track_ids[current_index]

    # Show the Spotify player and song name
    st.subheader("Is this the song?")
    spotify_player(current_song_id)
    st.write(f"Song Name: {current_song_name}")

    # Here, you can find another song
    if st.button("No, show me the next one."):
        # Move to the next song
        if current_index < len(st.session_state.track_ids) - 1:
            st.session_state.current_song_index += 1
        else:
            st.warning("No more songs to display!")


    if current_song_name in bb_df['title'].values:
        # Get random row and column indices
        random_row = np.random.randint(0, bb_df.shape[0])
        random_title = bb_df.iloc[random_row, 0]
        random_artist = bb_df.iloc[random_row, 1]
        spotify_search2 = sp.search(q=random_title + ' ' + random_artist, limit=1)
        hot_rec_id = spotify_search2['tracks']['items'][0]['id']
        spotify_player(hot_rec_id)
        st.write(f'Your song is currently sizzling hot, so here is another hot song: {random_title} by {random_artist}')
        # End this specific branch, but continue the app
        st.stop()

    else:
        st.write("Your song is not in the Billboard Top 100, but of course that doesnt mean anything. Here is a recommendation for you:")

    # get audio features of the input song
    af=pd.DataFrame(sp.audio_features(current_song_id))
    af_relevant=af[["danceability","energy","loudness","speechiness","acousticness",
        "instrumentalness","liveness","valence","tempo"]]

    # Load the scaler and KMeans model
    with open('scaler.pkl', 'rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)

    with open('kmeans_21_cluster.pkl', 'rb') as kmeans_file:
        loaded_kmeans = pickle.load(kmeans_file)

    af_scaled = loaded_scaler.transform(af_relevant)

    cluster_predicted = loaded_kmeans.predict(af_scaled)

    afeatures_df = pd.read_csv('afeatures_with_clusters.csv')

    cluster_filtered = afeatures_df[afeatures_df['cluster'] == int(cluster_predicted)]

    random_row = np.random.randint(0, cluster_filtered.shape[0])
    recommendation_id = cluster_filtered.iloc[random_row, 12]

    spotify_player(recommendation_id)
    # add genre info
    songs_genre_df = pd.read_csv('songs_genre.csv')
    genre_of_rec = songs_genre_df[songs_genre_df['id'] == recommendation_id]['genre'].values[0]
    st.write(f"The song's genre is {genre_of_rec} in case you wondered.")