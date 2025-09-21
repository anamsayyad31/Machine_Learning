# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 09:24:11 2025

@author: anams
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#load csv file
file_path = 'C:/7-recommender-systems/Entertainment.csv'
data = pd.read_csv(file_path)

#1. preprocess 'category' col using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
#remove common stop words
tfidf_matrix = tfidf.fit_transform(data['Category'])
#fit & transform cat data

#2. compute func cosine similarity betw titles
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#3. compute func to rec titles based on similarity
def get_recommendations(title, cosine_sim=cosine_sim):
    idx = data[data['Titles'] == title].index[0]
    
    #get pairwise similarity scores for all titles with that title
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    #sort by titles by sim scores in desc order
    sim_scores = sorted(sim_scores, key=lambda x : x[1], reverse=True)
       
    #get most sim title indices
    sim_indices = [i[0] for i in sim_scores[1:6]]
    #exclude 1st as its title itself
    
    #ret top 5 most sim. titles
    return data['Titles'].iloc[sim_indices]

#test rec system with eg. title
eg = "Toy Story (1995)"
rec = get_recommendations(eg)

#print recs
print(f'rec for {eg} :')
for title in rec :
    print(title)
