
import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
import sklearn
import category_encoders
import joblib
import ast
st.set_page_config(layout='wide')
#####################################################################################
# Cleaning
data = pd.read_csv('movies_metadata.csv', low_memory=False)
df = pd.DataFrame(data)
df = df.drop(['id' , 'budget' , 'overview', 'title','homepage','video' , 'imdb_id' ,'adult', 'tagline' , 'belongs_to_collection','poster_path'], axis=1)
df.rename(columns={'original_title': 'title'}, inplace=True)
df = df.drop_duplicates(subset='title', keep='first')
df.drop_duplicates(inplace=True)
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_date'] = df['release_date'].dt.year
df['genres_parsed'] = df['genres'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

df['type_of_movie'] = df['genres_parsed'].apply(
    lambda x: [d['name'] for d in x] if isinstance(x, list) else None
)
df['type_of_movie'] = df['type_of_movie'].apply(lambda x: x[0] if isinstance(x, list) and x else None)


df['production_companies_parsed'] = df['production_companies'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

df['company_names'] = df['production_companies_parsed'].apply(
    lambda x: [d['name'] for d in x] if isinstance(x, list) else None
)
df['company_names'] = df['company_names'].apply(lambda x: x[0] if isinstance(x, list) and x else None)


df['spoken_languages_parsed'] = df['spoken_languages'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

df['languages'] = df['spoken_languages_parsed'].apply(
    lambda x: [d['name'] for d in x] if isinstance(x, list) else None
)
df['languages'] = df['languages'].apply(lambda x: x[0] if isinstance(x, list) and x else None)


df['production_countries_parsed'] = df['production_countries'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

df['countries'] = df['production_countries_parsed'].apply(
    lambda x: [d['name'] for d in x] if isinstance(x, list) else None
)
df['countries'] = df['countries'].apply(lambda x: x[0] if isinstance(x, list) and x else None)

df = df.drop(['original_language' , 'production_companies' ,'production_countries' ,
              'production_companies_parsed' , 'spoken_languages_parsed' ,
              'production_countries_parsed' ,'spoken_languages' , 'genres_parsed' , 'genres'] , axis=1) 


df['countries'] = df['countries'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
df['languages'] = df['languages'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
df['company_names'] = df['company_names'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
df['type_of_movie'] = df['type_of_movie'].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)
df['countries'] = pd.DataFrame(df['countries'])
df['languages'] = pd.DataFrame(df['languages'])
df['company_names'] = pd.DataFrame(df['company_names'])
df['type_of_movie'] = pd.DataFrame(df['type_of_movie'])
df.replace('', np.nan, inplace=True)

df['runtime'] = df['runtime'].apply(lambda x: df['runtime'].mean() if pd.isna(x) else x)
df['status'] = df['status'].apply(lambda x: df['status'].mode()[0] if pd.isna(x) else x)
df['revenue'] = df['revenue'].apply(lambda x: df['revenue'].mean() if pd.isna(x) else x)
df['vote_count'] = df['vote_count'].apply(lambda x: df['vote_count'].mean() if pd.isna(x) else x)
df['vote_average'] = df['vote_average'].apply(lambda x: df['vote_average'].mean() if pd.isna(x) else x)
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
df['popularity'] = df['popularity'].apply(lambda x: df['popularity'].mean() if pd.isna(x) else x)
df['type_of_movie'] = df['type_of_movie'].apply(lambda x: df['type_of_movie'].mode()[0] if pd.isna(x) else x)
df['popularity'] = df['popularity'].astype(int)
df['release_date']= df['release_date'].apply(lambda x: df['release_date'].mode()[0] if pd.isna(x) else x)

df['type_of_movie'] = df['type_of_movie'].apply(
    lambda x: df['type_of_movie'].mode()[0] if pd.isna(x) else x)

df['languages'] = df['languages'].apply(
    lambda x: df['languages'].mode()[0] if pd.isna(x) else x)

df['countries'] = df['countries'].apply(
    lambda x: df['countries'].mode()[0] if pd.isna(x) else x)

df['company_names'] = df['company_names'].apply(
    lambda x: df['company_names'].mode()[0] if pd.isna(x) else x)
df_lst_true = ['Action' , 'Horror' , 'Crime' , 'Thriller' , 'Western' , 'War' , 'Mystery' , 'Romance']
adult_t = []
for col in df['type_of_movie']:
    if col in df_lst_true:
        adult_t.append(1)
    else:
        adult_t.append(0)
df['adult'] = adult_t
def is_adult(genres):
    if isinstance(genres, str):
        genres = [g.strip() for g in genres.split(',')]
    return int(any(genre in df_lst_true for genre in genres))

df['adult'] = df['type_of_movie'].apply(is_adult)
################################################################################################
## -movie name and revnu top 20
st.title('Movie Name and Revnu Top 20')
rev_ = df.groupby(['title'])['revenue'].sum().reset_index()
rev_ = rev_.sort_values(by ='revenue' ,ascending=0).head(20)
q1 = px.histogram(rev_ , x='title', y = 'revenue' , text_auto=True)
st.plotly_chart(q1)
    #type of movie are most common
st.title('type of movie are most common')
type_counts = df.groupby(['type_of_movie'])['vote_count'].sum().reset_index()
type_counts = type_counts.sort_values(by='vote_count' , ascending=1)
q2 = px.bar(type_counts , x='type_of_movie', y = 'vote_count')
st.plotly_chart(q2)

    #top 20 countries produce the most movies
st.title('Top 20 countries produce the most movies')
countries_counts = df['countries'].value_counts().reset_index().head(20)
countries_counts.columns = ['countries', 'count']
q3 = px.bar(countries_counts, x='countries', y='count')
st.plotly_chart(q3)

country_counts = df['countries'].value_counts().reset_index()
country_counts.columns = ['countries', 'movie_count']
q33 = px.choropleth(
country_counts,
locations='countries',
locationmode='country names',
color='movie_count',
hover_name='countries',
color_continuous_scale='Blues',
title='Movies Produced by Country')
st.plotly_chart(q33)

    #languages are most commonly used in movies
st.title('Languages are most commonly used in movies')
languag = df['languages'].value_counts().reset_index().head(20)
languag.columns = ['languages', 'languages_count']
q4 = px.bar(languag , x='languages', y = 'languages_count')
st.plotly_chart(q4)
    #companies produce the most high-rated movies
st.title('Companies produce the most high-rated movies')
q5_filter = df.groupby(['countries', 'company_names'])['vote_average'].sum().reset_index()
q5_filter = q5_filter[q5_filter['countries'] == 'United States of America']
q5_filter = q5_filter.sort_values(by='vote_average', ascending=0).head(5)
q5 = px.bar(q5_filter, x='vote_average', y='company_names',color='vote_average')
st.plotly_chart(q5)
    #the top 10 highest-rated movies
st.title('The top 10 highest-rated movies')
top_mov = df.groupby(['title'])['vote_average'].sum().reset_index()
top_mov = top_mov.sort_values(by = 'vote_average' , ascending =0).head(10)
q6 =px.bar(top_mov , x ='title', y='vote_average')
st.plotly_chart(q6)
    #production companies (company_names) have the highest total votes or average ratings
st.title('Production companies have the highest total average ratings')
q7_filter = df.groupby('company_names')['vote_count'].sum().reset_index()
q7_filter = q7_filter.sort_values(by='vote_count', ascending=False).head(5)
q7_filter['highlight'] = q7_filter['company_names'] == q7_filter.iloc[0]['company_names']
q7 = px.pie(q7_filter,
                 names='company_names',
                 values='vote_count',
                 title='Top 5 Movie Types by Vote Count',
                 color='highlight',
                 color_discrete_map={True: 'gold', False: 'lightgray'},
                 hole=0.2)
q7.update_traces(textposition='inside', textinfo='percent+label')
st.plotly_chart(q7)
    #average runtime of movies across different
st.title('Average runtime of movies across different')
q8_filter = df.groupby(['type_of_movie'])['runtime'].sum().reset_index()
q8_filter = q8_filter.sort_values(by='runtime' , ascending=0).head(10)
q8 = px.histogram(q8_filter , x='type_of_movie' , y ='runtime')
st.plotly_chart(q8)
    #Trend of Movie Revenue by Release Date
st.title('Trend of Movie Revenue by Release Date')
q99 = df.groupby('release_date')['revenue'].sum().reset_index()
q99 = q99.sort_values(by='revenue', ascending=True)
q9 = px.line(q99 , x='revenue' , y ='release_date')
st.plotly_chart(q9)
