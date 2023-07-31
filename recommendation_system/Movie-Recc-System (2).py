

import numpy as np
import pandas as pd
import ast




movies = pd.read_csv('tmdb_5000_movies.csv')
movies2 = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')





movies = movies.merge(credits,on='title')




movies2 = movies2.merge(credits,on='title')




# genres
# id
# keywords
# title
# overview
# cast
# crew
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]



movies2 = movies2[['movie_id','title','overview','genres','keywords']]



movies.isnull().sum()


# In[109]:


movies2.isnull().sum()


# In[110]:


movies.dropna(inplace = True)


# In[111]:


movies2.dropna(inplace = True)


# In[112]:


movies.duplicated().sum()


# In[113]:


movies2.duplicated().sum()






def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L



movies['genres'] = movies['genres'].apply(convert)





movies2['genres'] = movies2['genres'].apply(convert)







def convert3(obj):
    L = []
    counter = 0 
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L




movies['cast'] = movies['cast'].apply(convert3)





def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L





movies['crew'] = movies['crew'].apply(fetch_director)





movies.head()





movies['overview']=movies['overview'].apply(lambda x:x.split())





movies.head()





movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])



movies.head()





movies['tags'] = movies['overview'] + movies['keywords'] + movies['genres'] + movies['cast'] + movies['crew']




movies.head()




new_df = movies[['movie_id','title','tags']]




new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))




new_df.head()





new_df['tags'][0]




new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())





new_df.head()



get_ipython().system('pip install scikit_learn')




from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')





vectors = cv.fit_transform(new_df['tags']).toarray()



vectors[0]





cv.get_feature_names_out()




get_ipython().system('pip install nltk')





import nltk



from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()



def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)




new_df['tags'] = new_df['tags'].apply(stem)





cv.get_feature_names_out()




from sklearn.metrics.pairwise import cosine_similarity





similarity = cosine_similarity(vectors)




similarity[0]





def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
        


# In[156]:


recommend('Batman Begins')




def filter_movies_by_genre(movies2, genres, count=5):
    filtered_movies = movies2[movies2['genres'].apply(lambda x: any(item for item in genres if item in x))]
    return filtered_movies['title'].head(count)




genres = ['Action','Adventure']




filtered_movies = filter_movies_by_genre(movies2,genres, 5)



print(filtered_movies)






