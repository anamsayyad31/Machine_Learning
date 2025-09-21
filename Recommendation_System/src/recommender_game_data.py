# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 09:54:59 2025

@author: anams
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
#load data
data=pd.read_csv('C:/7-recommender-systems/game.csv')

#clean game names to avoid mismatches
data= data.copy()
data['game'] = (
    data['game'].astype(str).str.strip().str.replace(r'\s+', ' ',regex=True) )
'''
data= data.copy()
makes copy of df so that we dont modify org. one by accident

data['game'].astype(str)
ensures that 'game' col is treated as text.
somtimes, data load with mixed types(no. , nans) , so we convert evrything to str

str.strip()
removes any extra space at start or end of game
eg. ' halo ' = 'halo'
str.replace(r'\s+', ' ',regex=True)
replace multiple spaces (double/triple spaces, tabs) with single space
eg. 'call od duty    ' = 'call od duty'
'''
#pivot (mean if duplicate userId + game)
user_item_matrix = data.pivot_table(index='userId', columns='game', values='rating',aggfunc='mean')

#if user1 has 2 rating 
user_item_matrix = user_item_matrix.sort_index(axis=1)
user_item_matrix_filled = user_item_matrix.fillna(0)

#item-item cosine sim.
item_similarity = cosine_similarity(user_item_matrix_filled.T)
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix_filled.columns,
                                  columns=user_item_matrix_filled.columns)

#pop (global mean rating)
game_pop = data.groupby('game')['rating'].mean()
popularity_norm = game_pop / game_pop.max()

def get_hybrid_rec(user_id, num_recs=5, alpha=0.7):
    if user_id not in user_item_matrix_filled.index:
        raise ValueError(f'user_id {user_id} not found in data.')
        
    user_ratings = user_item_matrix_filled.loc[user_id]

    rated_games = user_ratings[user_ratings > 0].index.tolist()
    
    weighted_scores = np.zeros(len(item_similarity_df), dtype=float)
    
    for game in rated_games:
        if game not in item_similarity_df.columns:
            continue
        sim_vec = item_similarity_df[game].values
        weighted_scores += user_ratings[game] * sim_vec
        

    #convert to series
    item_scores = pd.Series(weighted_scores, index=item_similarity_df.index)
    
    item_scores = item_scores.drop(rated_games, errors='ignore')
    
    if item_scores.max() > 0:
        item_scores_norm = item_scores / item_scores.max()
    else:
        item_scores_norm = item_scores
    #normalize by dividing by max
    #only if max is +ve.this keeps scale comparable to pop_norm for clean blend.   
    #align pop.    
    pop_aligned = popularity_norm.reindex(item_scores_norm.index).fillna(0)
    #make sure pop. vector has exactly same items in same order as item_scores_norm. missing item get 0 pop.
    #blend scores
    final_scores = alpha * item_scores_norm + (1 - alpha) * pop_aligned
    #hybrid score = convex combo. of (norm. item-based score) & (pop)
    #return top-N
    return final_scores.sort_values(ascending=False).head(num_recs)

print('hybrid recs for user 3: ')
print(get_hybrid_rec(user_id=3, num_recs=5, alpha=0.7))    
   
#assignment: find dataset of game(like imbd), profession rather than software engineering rec     