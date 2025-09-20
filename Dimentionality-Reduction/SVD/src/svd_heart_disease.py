# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 16:38:04 2025

@author: anams
"""
import pandas as pd
import numpy as np
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

labels6, sil6, db6, ch6 = evaluate_clustering(df)

# Print results
print(f"  Silhouette Score:        {sil6:.4f}") #0.0923
print(f"  Davies-Bouldin Score:    {db6:.4f}") #2.5396
print(f"  Calinski-Harabasz Score: {ch6:.2f}\n") #28.05

#perform svd
import pandas as pd
data = pd.read_csv('C:/5-python_crisp-ml(q)/understanding_data_assignments_datasets/heart disease_prep.csv') 
data.head()
from sklearn.decomposition import TruncatedSVD
svd=TruncatedSVD(n_components=3)
svd.fit(data)
result=pd.DataFrame(svd.transform(data))
result.head()
result.columns='pc0','pc1','pc2'
result.head()

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram

# Dendrogram for svd 
z = linkage(result, method='complete', metric='euclidean')
plt.figure(figsize=(12, 6))
plt.title('Dendrogram (PCA with 6 components)')
dendrogram(z, leaf_rotation=90., leaf_font_size=10.)
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.tight_layout()
plt.show() #5

#clustering after svd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_clustering(X_reduced, n_clusters=5):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete', metric='euclidean')
    labels = model.fit_predict(X_reduced)

    sil = silhouette_score(X_reduced, labels)
    db = davies_bouldin_score(X_reduced, labels)
    ch = calinski_harabasz_score(X_reduced, labels)

    return labels, sil, db, ch

labels6, sil6, db6, ch6 = evaluate_clustering(result)

# Print results
print(f"  Silhouette Score:        {sil6:.4f}") #0.2843
print(f"  Davies-Bouldin Score:    {db6:.4f}") #1.0215
print(f"  Calinski-Harabasz Score: {ch6:.2f}\n") #190.00

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
score(0.2843):
moderate clustering quality.
clusters show better separation and compactness.
significant improvement over previous silhouette score.

score(1.0215):
noticeable improvement in clustering structure.
clusters are tighter and more distinct.
reflects better internal consistency.

score(190.00):
strong clustering strength.
clearer groupings with improved separation.
suggests better model tuning and cluster configuration.
'''
