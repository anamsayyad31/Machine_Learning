# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 16:44:49 2025

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
df = pd.read_csv('C:/8-PCA,SVD/wine_data_prep.csv')  # or use pd.DataFrame if already in memory

# Drop target column before PCA
df = df.drop(['Type_2','Type_3'], axis=1)

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
print("clustering (5):")
print(f"  Silhouette Score:        {sil6:.4f}") #0.1584
print(f"  Davies-Bouldin Score:    {db6:.4f}") #1.6377
print(f"  Calinski-Harabasz Score: {ch6:.2f}\n") #42.10

#perform svd
import pandas as pd
data = pd.read_csv('C:/8-PCA,SVD/wine_data_prep.csv')
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
plt.show() 

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
print("after svd (5 clusters):")
print(f"  Silhouette Score:        {sil6:.4f}") #0.5866
print(f"  Davies-Bouldin Score:    {db6:.4f}") #0.7387
print(f"  Calinski-Harabasz Score: {ch6:.2f}\n") #706.17

'''
Previous Scores:
score(0.1584):
moderate clustering quality.
some separation between clusters, but still overlapping.
could benefit from tuning cluster count or algorithm.

score(1.6377):
average clustering structure.
clusters are not very tight or well-separated.
room for improvement with better features or linkage methods.

score(42.10):
low clustering strength.
grouping is weak with minimal separation.
suggests suboptimal cluster configuration.


Current Scores:
score(0.5866):
high clustering quality.
clusters are well separated and compact.
strong indication of optimal clustering.

score(0.7387):
good clustering structure.
clusters are tight and well-separated.
reflects strong internal consistency.

score(706.17):
excellent clustering strength.
clear and distinct groupings with strong separation.
model and cluster count likely well-tuned.
'''
