# -*- coding: utf-8 -*-
"""
Created on Fri Aug 29 17:11:48 2025

@author: anams
"""
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

# Load your preprocessed heart disease data
df = pd.read_csv('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/heart disease_prep.csv')  # or use pd.DataFrame if already in memory

# Drop target column before PCA
df = df.drop('target', axis=1)

z = linkage(df,method='complete',metric='euclidean')
plt.figure(figsize=(15,8));
plt.title('hierarchical clustering dendogram')
plt.xlabel('Index');
plt.ylabel('Distance');
#ref help of dendogram
#sch.dendrogram(z)
sch.dendrogram(z,leaf_rotation=40,leaf_font_size=10)
plt.show()

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_clustering(X_reduced, n_clusters=5):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete', metric='euclidean')
    labels = model.fit_predict(X_reduced)

    sil = silhouette_score(X_reduced, labels)
    db = davies_bouldin_score(X_reduced, labels)
    ch = calinski_harabasz_score(X_reduced, labels)

    return labels, sil, db, ch

# Evaluate PCA with 6 components
labels6, sil6, db6, ch6 = evaluate_clustering(df)

# Print results
print(f"  Silhouette Score:        {sil6:.4f}") #0.0923
print(f"  Davies-Bouldin Score:    {db6:.4f}") #2.5396
print(f"  Calinski-Harabasz Score: {ch6:.2f}\n") #28.05
'''
score(0.0923):
low clustering quality.
clusters are poorly separated or overlapping.
likely that the number of clusters is not optimal.

score(2.5396):
weak clustering structure.
clusters are loosely packed and not well separated.
consider tuning clustering parameters or trying different algorithms.


score(28.05):
low clustering strength.
grouping is weak with minimal separation.
may benefit from increasing cluster count or refining features.
'''


# Load your preprocessed heart disease data
df = pd.read_csv('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/heart disease_prep.csv')  # or use pd.DataFrame if already in memory

# Drop target column before PCA
X = df.drop('target', axis=1)
y = df['target'] 

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X)

# Explained variance
explained_var = pca.explained_variance_ratio_
cum_var = np.cumsum(explained_var)

# Plot cumulative explained variance
plt.figure(figsize=(8,5))
plt.plot(cum_var, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid(True)
plt.show() 

# Optional: print optimal number of components for ≥90% variance
optimal_components = np.argmax(cum_var >= 0.90) + 1
print(f"Optimal number of components: {optimal_components}") #9

plt.figure(figsize=(8,5))
plt.plot(explained_var, marker='o')
plt.xlabel('Component')
plt.ylabel('Explained Variance')
plt.title('Scree Plot')
plt.grid(True)
plt.show() #7

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# PCA with 6 components
pca6 = PCA(n_components=6).fit_transform(X)


from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram

# Dendrogram for PCA with 6 components
z6 = linkage(pca6, method='complete', metric='euclidean')
plt.figure(figsize=(12, 6))
plt.title('Dendrogram (PCA with 6 components)')
dendrogram(z6, leaf_rotation=90., leaf_font_size=10.)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.tight_layout()
plt.show() #5

#perform clustering after pca & compare results

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_clustering(X_reduced, n_clusters=5):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete', metric='euclidean')
    labels = model.fit_predict(X_reduced)

    sil = silhouette_score(X_reduced, labels)
    db = davies_bouldin_score(X_reduced, labels)
    ch = calinski_harabasz_score(X_reduced, labels)

    return labels, sil, db, ch

labels6, sil6, db6, ch6 = evaluate_clustering(pca6)

# Print results
print("PCA (6 components):")
print(f"  Silhouette Score:        {sil6:.4f}") #0.0892
print(f"  Davies-Bouldin Score:    {db6:.4f}") #2.2700
print(f"  Calinski-Harabasz Score: {ch6:.2f}\n") #39.29

'''
Previous Scores:
score(0.0923):
low clustering quality.
clusters are poorly separated or overlapping.
likely that the number of clusters is not optimal.

score(2.5396):
weak clustering structure.
clusters are loosely packed and not well separated.
consider tuning clustering parameters or trying different algorithms.

score(28.05):
low clustering strength.
grouping is weak with minimal separation.
may benefit from increasing cluster count or refining features.


Current Scores:
score(0.0892):
very low clustering quality.
clusters are poorly separated or overlapping.
slightly worse than previous silhouette score.

score(2.2700):
still weak clustering structure.
slightly better than previous DB score, but still high.
clusters remain loosely packed.

score(39.29):
slightly improved clustering strength.
better separation than before, but still not strong.
may benefit from more refined features or cluster tuning.
'''
