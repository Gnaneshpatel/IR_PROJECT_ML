import random
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from preprocess import *

st.sidebar.header("What's on top of your mind!!")


def user_input():
    text = st.sidebar.text_input('Write your thought! ')

    return text


input = user_input()

# st.write(input)
encoder = pickle.load(open('encoder.pkl', 'rb'))
cv = pickle.load(open('CountVectorizer.pkl', 'rb'))


model = tf.keras.models.load_model('my_model.h5')
input = preprocess(input)

array = cv.transform([input]).toarray()

pred = model.predict(array)
a = np.argmax(pred, axis=1)
prediction = encoder.inverse_transform(a)[0]
print(prediction)

mood = ""
if input == '':
    st.write('')
else:
    mood = prediction

st.header('Movie Recommender System')

# st.subheader("Recommendation on the basis of content")


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=208a5adae34de4dc409b5c6950954ff7&language=en-US".format(
        movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters


movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(
    open('similarity.pkl', 'rb'))

movie_list = movies['title'].values
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0])
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1])

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2])
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3])
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4])
    st.write("These movies are similar to the movie you selected and will help you have a satisfying movie-watching experience.")

st.subheader("Recommendation on the basis of mood")

data = pd.read_csv(
    'tmdb_5000_movies.csv')
df12 = pd.DataFrame(data)
data1 = data.loc[:, 'genres']

for i in range(len(data1)):
    string = data1[i]
    list_of_dicts = json.loads(string)
    data1[i] = list_of_dicts

df = pd.DataFrame(data1)
genres_list = data["genres"].apply(lambda x: [y["name"] for y in x]).explode()
# genres_list.unique()
df["genres"] = df["genres"].apply(lambda x: [y.get("name") for y in x])
df1 = pd.DataFrame(df["genres"].values.tolist())
df1.columns = ["name_{}".format(x) for x in range(len(df1.columns))]
df12['genres'] = df['genres']

df1 = pd.concat([df12[["id"]], df1], axis=1)
df1 = pd.concat([df12[["title"]], df1], axis=1)

df = pd.concat([df12[["vote_average"]], df1], axis=1)
df = df.melt(id_vars=["vote_average",  "title", "id"], value_vars=df.columns[1:],
             value_name="name")[["vote_average", "title", "id", "name"]].dropna()

rules = {
    "Drama": "because we think you are happy and might enjoy watching thought-provoking and emotional stories that may resonate with you on a deeper level.",
    "War": "because we think your mood is low and sometimes watching a thrilling and action-packed movie can provide an escape from reality and help shift the focus from the source of sadness, even if only temporarily.",
    "Comedy": "because comedy movies are a great choice to watch when you're feeling surprised because they can help you laugh and relieve any tension or shock you may be feeling. Laughter has been shown to release endorphins, which can boost your mood and help you feel more relaxed.",
    "Action": "because we think you are a little afraid right now and action movies can be a good choice to watch when you're feeling scared because they can help distract you from your fears and provide a sense of excitement and adrenaline. Watching intense fight scenes and daring stunts can also help you feel empowered and courageous."   
}

def suggest_movie(genre, mood):
    
    recommended_movie_posters = []
    # Filter the movies by genre and mood
    genre_data = df[df['name'] == genre]
    # print(genre_data)
    mood_data = genre_data[genre_data['name'] == mood]
    # print(mood_data)
    # Sort the movies by rating in descending order
    sorted_data = mood_data.sort_values('vote_average', ascending=False)
    # print(sorted_data['title'].loc[:10])
    # Return the top 10 movie titles
    # print(sorted_data['title'])
    explanation = "We recommend these movies based on your previous choices."
    if genre in rules.keys():
        explanation = f"We recommend these movies {rules[genre]}"
    top_movies1 = sorted_data['title'].iloc[:15].tolist()
    top_movies_id1 = sorted_data['id'].iloc[:15].tolist()
    top_movies = random.sample(top_movies1, 5)
    top_movies_id = [id for val, id in zip(
        top_movies1, top_movies_id1) if val in top_movies]
    for i in top_movies_id:
        recommended_movie_posters.append(fetch_poster(i))
    return top_movies, recommended_movie_posters, explanation


# col1, col2, col3, col4, col5 = st.columns(5)

# with col1:
#     button1 = st.button("Happy Mood😃")

# with col2:
#     button2 = st.button('Sad Mood😞')

# with col3:
#     button3 = st.button('Chill Mood🤩')

# with col4:
#     button4 = st.button('Adventurous Mood🎬')
# with col5:
#     button5 = st.button('Romantic Mood😍')


mood1 = ""
if mood == "joy":
    mood1 = "Drama"
    st.button("It seems you are Happy!!😃")
    romantic_movies, romantic_movies_posters, explanation = suggest_movie(mood1, mood1)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(romantic_movies[0])
        st.image(romantic_movies_posters[0])
    with col2:
        st.text(romantic_movies[1])
        st.image(romantic_movies_posters[1])

    with col3:
        st.text(romantic_movies[2])
        st.image(romantic_movies_posters[2])
    with col4:
        st.text(romantic_movies[3])
        st.image(romantic_movies_posters[3])
    with col5:
        st.text(romantic_movies[4])
        st.image(romantic_movies_posters[4])
    # st.write(explanation)
elif mood == "sadness":
    mood1 = "War"
    st.button("It seems you are sad!!😃")
    romantic_movies, romantic_movies_posters, explanation = suggest_movie(mood1, mood1)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(romantic_movies[0])
        st.image(romantic_movies_posters[0])
    with col2:
        st.text(romantic_movies[1])
        st.image(romantic_movies_posters[1])

    with col3:
        st.text(romantic_movies[2])
        st.image(romantic_movies_posters[2])
    with col4:
        st.text(romantic_movies[3])
        st.image(romantic_movies_posters[3])
    with col5:
        st.text(romantic_movies[4])
        st.image(romantic_movies_posters[4])
    # st.write(explanation)
elif mood == "surprised":
    mood1 = "Comedy"
    st.button("It seems you are surprised!!😃")
    romantic_movies, romantic_movies_posters, explanation = suggest_movie(mood1, mood1)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(romantic_movies[0])
        st.image(romantic_movies_posters[0])
    with col2:
        st.text(romantic_movies[1])
        st.image(romantic_movies_posters[1])

    with col3:
        st.text(romantic_movies[2])
        st.image(romantic_movies_posters[2])
    with col4:
        st.text(romantic_movies[3])
        st.image(romantic_movies_posters[3])
    with col5:
        st.text(romantic_movies[4])
        st.image(romantic_movies_posters[4])
    # st.write(explanation)
elif mood == "fear":
    mood1 = "Action"
    st.button("It seems you are afraid!!😃")
    romantic_movies, romantic_movies_posters, explanation = suggest_movie(mood1, mood1)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(romantic_movies[0])
        st.image(romantic_movies_posters[0])
    with col2:
        st.text(romantic_movies[1])
        st.image(romantic_movies_posters[1])

    with col3:
        st.text(romantic_movies[2])
        st.image(romantic_movies_posters[2])
    with col4:
        st.text(romantic_movies[3])
        st.image(romantic_movies_posters[3])
    with col5:
        st.text(romantic_movies[4])
        st.image(romantic_movies_posters[4])
    # st.write(explanation)
elif mood == "love":
    mood1 = "Romance"
    st.button("It seems you are in love!!😃")
    romantic_movies, romantic_movies_posters, explanation = suggest_movie(mood1, mood1)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(romantic_movies[0])
        st.image(romantic_movies_posters[0])
    with col2:
        st.text(romantic_movies[1])
        st.image(romantic_movies_posters[1])

    with col3:
        st.text(romantic_movies[2])
        st.image(romantic_movies_posters[2])
    with col4:
        st.text(romantic_movies[3])
        st.image(romantic_movies_posters[3])
    with col5:
        st.text(romantic_movies[4])
        st.image(romantic_movies_posters[4])
    # st.write("We recommend these movies because we think you are in love or feeling loved and a romantic movie is a perfect choice to watch when you're feeling in love because it can amplify and enhance those positive feelings. Romantic movies often feature heartwarming stories and can remind you of the beauty of love, making you feel more connected to those emotions.")
elif mood == "anger":
    mood1 = "Romance"
    st.button("It seems you are angry!!😃")
    romantic_movies, romantic_movies_posters, explanation = suggest_movie(mood1, mood1)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(romantic_movies[0])
        st.image(romantic_movies_posters[0])
    with col2:
        st.text(romantic_movies[1])
        st.image(romantic_movies_posters[1])

    with col3:
        st.text(romantic_movies[2])
        st.image(romantic_movies_posters[2])
    with col4:
        st.text(romantic_movies[3])
        st.image(romantic_movies_posters[3])
    with col5:
        st.text(romantic_movies[4])
        st.image(romantic_movies_posters[4])
    # st.write("We recommend these movies because we think you are angry and when feeling angry, watching a romantic movie can help you relax and shift your focus away from negative emotions. Romantic movies typically feature pleasant music, beautiful settings, and happy endings, which can help to elevate your mood and soothe your feelings of anger.")
