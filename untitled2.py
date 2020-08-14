# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:43:31 2020

@author: 91984
"""
from surprise import Dataset
from surprise import Reader
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd
from sklearn.metrics.pairwise import linear_kernel 
from surprise import SVD

md=pd.read_csv(r"C:/Users/91984/Downloads/movies.csv",header=0)
links_small=pd.read_csv(r"C:/Users/91984/Downloads/dataset/links_small.csv",header=0)
md['description']=md['description'].fillna('')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
md = md.drop([19730, 29503, 35587])
md['id'] = md['id'].astype('int')
smd = md[md['id'].isin(links_small)]
smd=smd.reset_index()
#%%
smd['keywords']=smd['keywords'].fillna("")
smd['cast']=smd['cast'].fillna("")
smd['director']=smd['director'].fillna("")
smd['soup']=smd['director']+smd['cast']+smd['keywords']
cv=CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
cv_matrix=cv.fit_transform(smd['soup'])
cosine_sim=linear_kernel(cv_matrix,cv_matrix)

#%%
reader=Reader()
svd=SVD()
ratings = pd.read_csv(r'C:/Users/91984/Downloads/dataset/ratings_small.csv',header=0)
data=Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)


data.split(n_folds=5)
train=data.build_full_trainset()
svd.fit(train)

#%%
indices = pd.Series(smd.index, index=smd['title'])

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
    return movies['title']
#%%
print(get_recommendations(1,'GoldenEye'))



    
    
    
















