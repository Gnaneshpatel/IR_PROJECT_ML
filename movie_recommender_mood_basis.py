import random
import json
import requests
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from preprocess import *
from streamlit_text_rating.st_text_rater import st_text_rater
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
#from streamlit_star_rating import st_star_rating
import ast


#------------------------------------------------------------------------------------------------------------------#

def top_10_movies():
    ########## Read datasets ################
    credit = pd.read_csv('tmdb_5000_credits.csv')
    movie = pd.read_csv('tmdb_5000_movies.csv')
    
    ############### merge 2 datasets ################
    merged= movie.merge(credit,left_on=['id','title'],right_on=['movie_id','title'])

    #Calculate Weighted Rating

    # movies which is higher than the mean can be in the chart
    C= merged['vote_average'].mean()  

    # movies voted have to be more than 90% percentile to enter the chart
    m= merged['vote_count'].quantile(0.9)

    #Create a copy file and filter the qualified movie
    q_movies = merged.copy().loc[merged['vote_count'] >= m]
    print(q_movies.shape)

    # There are 481 movies qualified to be in the chart

    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        # Calculation based on the IMDB formula
        return (v/(v+m) * R) + (m/(m+v) * C)

    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

    q_movies = q_movies.sort_values('score', ascending=False)

    #Print the top 10 movies
    top_10=q_movies[['id','title', 'vote_count','score']].head(10).values.tolist()
    poster_list=[]
    for i in top_10:
        ans=fetch_poster(i[0])
        poster_list.append(ans)
    return top_10,poster_list

#------------------------------------------------------------------------------------------------------------------#

# FOR LOCATION
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 

data_ = pd.read_csv(
    'tmdb_5000_movies.csv')
df12_ = pd.DataFrame(data_)
# print(data_)
data_['production_countries'] = data_['production_countries'].apply(convert)
c = []
for i in data_['production_countries']:
    for j in i:
        if j not in c:
            c.append(j)

# print(c)

st.sidebar.subheader("What's on top of your mind!!")


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


# st.subheader("Recommendation on the basis of content")


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=208a5adae34de4dc409b5c6950954ff7&language=en-US".format(
        movie_id)
    data = requests.get(url)
    data = data.json()
    poster_path = data['poster_path']
    full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
    return full_path


def recommend(movie, selected_location):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(
        list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances:
        # fetch the movie poster
        if len(recommended_movie_names)>=5 and len(recommended_movie_posters)>=5:
            break
        if(i[0]<len(data_)):
            if(selected_location in data_.iloc[i[0]].production_countries):
                movie_id = movies.iloc[i[0]].movie_id
                recommended_movie_posters.append(fetch_poster(movie_id))
                recommended_movie_names.append(movies.iloc[i[0]].title)
    return recommended_movie_names, recommended_movie_posters

def countries():
    c = []
    for i in movies["production_countries"]:
        for j in i:
            if j not in c:
                c.append(j)

def feedback():
    label = "User Rating"
    #stars = st_star_rating(label, 5,defaultValue=0, size=20)

##############################################################################################################

def colaborative():
    ratings = pd.read_csv('D:/Semester 2/Emotion-Detection-from-Text-using-Neural-Netwroks-main/ratings_small.csv')
    #Preparing matrix for user-based and item-based
    
    user_rating = ratings.pivot(index='userId', columns='movieId', values='rating')
    # user_rating
    avg_ratings = user_rating.mean(axis=1)
    # avg_ratings
    user_ratings_pivot = user_rating.sub(avg_ratings, axis=0)
    # # user_based matrix
    user_ratings_pivot.fillna(0, inplace=True)
    # # Change from user_based to item_based matrix
    movie_ratings_pivot = user_ratings_pivot.T
    
    #-----------------------------------------------------------
    
    similarities = cosine_similarity(movie_ratings_pivot)
    cosine_similarity_df = pd.DataFrame(similarities,columns=movie_ratings_pivot.index,index=movie_ratings_pivot.index)
    print(cosine_similarity_df.shape)
    return cosine_similarity_df
    #--------------------------------------------------------
    
def get_similar_movies(movieId,user_rating):
    cosine_similarity_df = colaborative()
    similar_score = cosine_similarity_df[movieId]*user_rating-2.5
    similar_score = similar_score.sort_values(ascending = False)
    scores = similar_score[1:6]
    movie_ids = scores.index.tolist()
    return movie_ids

#######################################################################################################################

st.header('Top Rated Movies By Users')
Rated_movies,posters = top_10_movies()
k=0
for i in range(0,10,5):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.write(f'<div style="line-height:1.2">{Rated_movies[k+0][1]}</div>', unsafe_allow_html=True, width=135)
        st.write(f'<div style="line-height:1.2"> Votes: {Rated_movies[k+0][2]}</div>', unsafe_allow_html=True, width=135 )
        st.write(f'<div style="line-height:1.2"> Score: {round(Rated_movies[k+0][3],2)}</div>',unsafe_allow_html=True, width=135)
        st.image(posters[k+0], width=135)
        
    with col2:
        st.write(f'<div style="line-height:1.2">{Rated_movies[k+1][1]}</div>', unsafe_allow_html=True)
        st.write(f'<div style="line-height:1.2"> Votes: {Rated_movies[k+1][2]}</div>', unsafe_allow_html=True)
        st.write(f'<div style="line-height:1.2"> Score: {round(Rated_movies[k+1][3],2)}</div>',unsafe_allow_html=True)
        st.image(posters[k+1], width=135)

    with col3:
        st.write(f'<div style="line-height:1.2">{Rated_movies[k+2][1]}</div>', unsafe_allow_html=True)
        st.write(f'<div style="line-height:1.2"> Votes: {Rated_movies[k+2][2]}</div>', unsafe_allow_html=True)
        st.write(f'<div style="line-height:1.2"> Score: {round(Rated_movies[k+2][3],2)}</div>',unsafe_allow_html=True)
        st.image(posters[k+2], width=135)
    with col4:
        st.write(f'<div style="line-height:1.2">{Rated_movies[k+3][1]}</div>', unsafe_allow_html=True)
        st.write(f'<div style="line-height:1.2"> Votes: {Rated_movies[k+3][2]}</div>', unsafe_allow_html=True)
        st.write(f'<div style="line-height:1.2"> Score: {round(Rated_movies[k+3][3],2)}</div>',unsafe_allow_html=True)
        st.image(posters[k+3], width=135)
    with col5:
        st.write(f'<div style="line-height:1.2">{Rated_movies[k+4][1]}</div>', unsafe_allow_html=True)
        st.write(f'<div style="line-height:1.2"> Votes: {Rated_movies[k+4][2]}</div>', unsafe_allow_html=True)
        st.write(f'<div style="line-height:1.2"> Score: {round(Rated_movies[k+4][3],2)}</div>',unsafe_allow_html=True)
        st.image(posters[k+4], width=135)
    k=k+5

###############################################################################################################

st.header('Movie Recommender System')

movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(
    open('similarity.pkl', 'rb'))

movie_list = movies['title'].values
selected_location = st.selectbox("Select your location",c )
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list
)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(
        selected_movie, selected_location)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if(len(recommended_movie_names)>0 and len(recommended_movie_posters)>0):
            st.text(recommended_movie_names[0])
            st.image(recommended_movie_posters[0], width=135)
        else:
            print("NO RECOMMENDATIONS")
    with col2:
        if(len(recommended_movie_names)>1 and len(recommended_movie_posters)>1):
            st.text(recommended_movie_names[1])
            st.image(recommended_movie_posters[1], width=135)
    with col3:
        if(len(recommended_movie_names)>2 and len(recommended_movie_posters)>2):
            st.text(recommended_movie_names[2])
            st.image(recommended_movie_posters[2], width=135)
    with col4:
        if(len(recommended_movie_names)>3 and len(recommended_movie_posters)>3):
            st.text(recommended_movie_names[3])
            st.image(recommended_movie_posters[3], width=135)
    with col5:
        if(len(recommended_movie_names)>4 and len(recommended_movie_posters)>4):
            st.text(recommended_movie_names[4])
            st.image(recommended_movie_posters[4], width=135)

    st.write("These movies are similar to the movie you selected and will help you have a satisfying movie-watching experience.")

    feedback()






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
    top_movies1 = sorted_data['title'].iloc[:15].tolist()
    top_movies_id1 = sorted_data['id'].iloc[:15].tolist()
    top_movies = random.sample(top_movies1, 5)
    top_movies_id = [id for val, id in zip(
        top_movies1, top_movies_id1) if val in top_movies]
    for i in top_movies_id:
        recommended_movie_posters.append(fetch_poster(i))
    return top_movies, recommended_movie_posters


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
    romantic_movies, romantic_movies_posters = suggest_movie(mood1, mood1)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(romantic_movies[0])
        st.image(romantic_movies_posters[0], width=135)
    with col2:
        st.text(romantic_movies[1])
        st.image(romantic_movies_posters[1], width=135)

    with col3:
        st.text(romantic_movies[2])
        st.image(romantic_movies_posters[2], width=135)
    with col4:
        st.text(romantic_movies[3])
        st.image(romantic_movies_posters[3], width=135)
    with col5:
        st.text(romantic_movies[4])
        st.image(romantic_movies_posters[4], width=135)
        
    feedback()
    

elif mood == "sadness":
    mood1 = "War"
    st.button("It seems you are sad!!😞")
    romantic_movies, romantic_movies_posters = suggest_movie(mood1, mood1)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(romantic_movies[0])
        st.image(romantic_movies_posters[0], width=135)
    with col2:
        st.text(romantic_movies[1])
        st.image(romantic_movies_posters[1], width=135)

    with col3:
        st.text(romantic_movies[2])
        st.image(romantic_movies_posters[2], width=135)
    with col4:
        st.text(romantic_movies[3])
        st.image(romantic_movies_posters[3], width=135)
    with col5:
        st.text(romantic_movies[4])
        st.image(romantic_movies_posters[4], width=135)
    feedback()
    
elif mood == "surprised":
    mood1 = "Comedy"
    st.button("It seems you are surprised!!😯")
    romantic_movies, romantic_movies_posters = suggest_movie(mood1, mood1)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(romantic_movies[0])
        st.image(romantic_movies_posters[0], width=135)
    with col2:
        st.text(romantic_movies[1])
        st.image(romantic_movies_posters[1], width=135)

    with col3:
        st.text(romantic_movies[2])
        st.image(romantic_movies_posters[2], width=135)
    with col4:
        st.text(romantic_movies[3])
        st.image(romantic_movies_posters[3], width=135)
    with col5:
        st.text(romantic_movies[4])
        st.image(romantic_movies_posters[4], width=135)
    feedback()
    
elif mood == "fear":
    mood1 = "Action"
    st.button("It seems you are scared!!😰")
    romantic_movies, romantic_movies_posters = suggest_movie(mood1, mood1)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(romantic_movies[0])
        st.image(romantic_movies_posters[0], width=135)
    with col2:
        st.text(romantic_movies[1])
        st.image(romantic_movies_posters[1], width=135)

    with col3:
        st.text(romantic_movies[2])
        st.image(romantic_movies_posters[2], width=135)
    with col4:
        st.text(romantic_movies[3])
        st.image(romantic_movies_posters[3], width=135)
    with col5:
        st.text(romantic_movies[4])
        st.image(romantic_movies_posters[4], width=135)
    feedback()
    
elif mood == "love":
    mood1 = "Romance"
    st.button("It seems you are feeling romantic!!😍")
    romantic_movies, romantic_movies_posters = suggest_movie(mood1, mood1)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(romantic_movies[0])
        st.image(romantic_movies_posters[0], width=135)
    with col2:
        st.text(romantic_movies[1])
        st.image(romantic_movies_posters[1], width=135)

    with col3:
        st.text(romantic_movies[2])
        st.image(romantic_movies_posters[2], width=135)
    with col4:
        st.text(romantic_movies[3])
        st.image(romantic_movies_posters[3], width=135)
    with col5:
        st.text(romantic_movies[4])
        st.image(romantic_movies_posters[4], width=135)
    feedback()
    
elif mood == "anger":
    mood1 = "Romance"
    st.button("It seems you are angry!!😠")
    romantic_movies, romantic_movies_posters = suggest_movie(mood1, mood1)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(romantic_movies[0])
        st.image(romantic_movies_posters[0], width=135)
    with col2:
        st.text(romantic_movies[1])
        st.image(romantic_movies_posters[1], width=135)

    with col3:
        st.text(romantic_movies[2])
        st.image(romantic_movies_posters[2], width=135)
    with col4:
        st.text(romantic_movies[3])
        st.image(romantic_movies_posters[3], width=135)
    with col5:
        st.text(romantic_movies[4])
        st.image(romantic_movies_posters[4], width=135)
    #feedback()



colab_key='colab'
st.header('Colaborative system: ')
selected_movie = st.selectbox(
    "Type or select a movie from the dropdown",
    movie_list,key=colab_key
)

st.header('Rate the choosen movie: ')
# Display a slider widget for the user to select their rating
rating = st.slider('Select your rating', min_value=0, max_value=5, step=1)

# Display the selected rating to the user
st.write('You selected a rating of', rating)

def colab_recommend(selected_movie,rating):
    index = movies[movies['title'] == selected_movie].index[0]
    ind= movies.iloc[index].movie_id
    movie_ids = get_similar_movies(ind, rating)
    recommended_movie_names = []
    recommended_movie_posters = []
    print(movie_ids)
    for i in movie_ids:
        recommended_movie_posters.append(fetch_poster(i))
        index = movies[movies['movie_id'] == i].index[0]
        recommended_movie_names.append(movies.iloc[index].title)
        
    return recommended_movie_names,recommended_movie_posters
    
    
buttun_key="heli"
if st.button('Show Recommendation',key = buttun_key):
    recommended_movie_names,recommended_movie_posters = colab_recommend(selected_movie,rating)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0], width=135)
    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1], width=135)

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2], width=135)
    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3], width=135)
    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4], width=135)





