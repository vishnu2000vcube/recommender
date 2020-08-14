# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:43:31 2020

@author: 91984
"""
from surprise import Dataset
from surprise import Reader
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
from sklearn.metrics.pairwise import linear_kernel 
from surprise import SVD

md=pd.read_csv(r"C:/Users/91984/Downloads/dataset/movies_metadata.csv",header=0)
links_small=pd.read_csv(r"C:/Users/91984/Downloads/dataset/links_small.csv",header=0)
md['overview']=md['overview'].fillna('')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
md = md.drop([19730, 29503, 35587])
md['id'] = md['id'].astype('int')
smd = md[md['id'].isin(links_small)]
smd=smd.reset_index()
#%%
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview']
smd['description'] = smd['description'].fillna('')
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tf_matrix = tf.fit_transform(smd['description'])


cosine_sim=linear_kernel(tf_matrix,tf_matrix)


indices = pd.Series(smd.index, index=smd['title'])

#%%
reader=Reader()
svd=SVD()
ratings = pd.read_csv(r'C:/Users/91984/Downloads/dataset/ratings_small.csv',header=0)
data=Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)


data.split(n_folds=5)
train=data.build_full_trainset()
svd.fit(train)

#%%
def get_sim_movieidx(title):
    idx=indices[title]
    sim_scores=list(enumerate(cosine_sim[idx]))
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
    sim_scores=sim_scores[1:31]
    midx=[i[0] for i in sim_scores]
    return midx
#%%
def get_recommendations(userId,title):
    midx=get_sim_movieidx(title)
    mid=smd['id'].iloc[midx]
    movies = smd.iloc[midx][['title','id']]
    l=[]
    for i in mid:
        k=svd.predict(userId,i).est
        l.append(k)
    movies['est']=l
    movies=movies[0:10]
    movies = movies.sort_values('est', ascending=False)
    return movies[['title','est']]
#%%
print(get_recommendations(2,'GoldenEye'))



    
    
    
















