import streamlit as st
import pickle
import requests
import numpy as np

# 1. Load Data and Models from Pickle Files
@st.cache_data
def load_files():
    movies = pickle.load(open('movie_list.pkl', 'rb'))
    cv = pickle.load(open('vectorizer.pkl', 'rb'))
    knn = pickle.load(open('knn.pkl', 'rb'))
    vectors = cv.transform(movies['tags']).toarray()
    return movies, knn, vectors

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US"
        data = requests.get(url).json()
        poster_path = data.get('poster_path', None)
        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return "https://via.placeholder.com/300x450?text=No+Image"
    except Exception as e:
        return "https://via.placeholder.com/300x450?text=Error"

def recommend(selected_movie, movies, knn, vectors):
    try:
        idx = movies[movies['title'].str.lower() == selected_movie.lower()].index[0]
    except IndexError:
        st.warning(f'"{selected_movie}" not found in database.')
        return [], []
    distances, indices = knn.kneighbors([vectors[idx]])
    recommended_titles = []
    recommended_ids = []
    for i in indices[0][1:]:  # skip the movie itself
        recommended_titles.append(movies.iloc[i].title)
        recommended_ids.append(int(movies.iloc[i].movie_id))
    return recommended_titles, recommended_ids

# Streamlit UI
st.markdown(
    """
    <h1 style='font-family: Georgia, serif; color: #FF5733;'>
        🎬 Movie Recommender System (KNN-ML)
    </h1>
    """,
    unsafe_allow_html=True
)

from PIL import Image  # already used in some versions

# Load and display the logo
logo = Image.open('logo.jpg')  # Replace with your actual file name
st.image(logo, width=250)  # Adjust width as needed


movies, knn, vectors = load_files()
movie_list = movies['title'].values

selected_movie = st.selectbox("Type or select a movie from the dropdown", movie_list)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_ids = recommend(selected_movie, movies, knn, vectors)
    if recommended_movie_names:
        cols = st.columns(5)
        for idx, col in enumerate(cols):
            with col:
                st.text(recommended_movie_names[idx])
                st.image(fetch_poster(recommended_movie_ids[idx]))
    else:
        st.info("No recommendations found. Try another movie.")

