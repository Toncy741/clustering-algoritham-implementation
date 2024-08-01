# clustering-algoritham-implementation
The project involves clustering analysis of a dataset using two popular unsupervised machine learning algorithms
K-Means Clustering and Hierarchical Clustering. The goal is to identify patterns and structures in the data

*first step 
Import the necessary libraries to handle data manipulation, visualization, and clustering algorithms.

*import numpy as np
*import pandas as pd
*import matplotlib.pyplot as plt
*import seaborn as sns
*from sklearn import datasets
*from sklearn.cluster import KMeans
*from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

Load the Iris dataset from the sklearn library.
Convert the dataset into a Pandas DataFrame for easier manipulation and analysis then Visualize Data:

Use Seaborn to create a pair plot of the dataset  This helps in understanding the distribution and relationships between the features
Apply K-Means Clustering:

Initialize the K-Means algorithm with the desired number of clusters (in this case, 3)
Fit the model to the dataset and predict the clusters for each sample.
Visualize the clusters using a scatter plot, with different colors representing different clusters after that 

Apply Hierarchical Clustering:

Use the linkage method from the scipy.cluster.hierarchy module to perform hierarchical clustering on the dataset.
Visualize the hierarchical clustering using a dendrogram.
Form flat clusters by cutting the dendrogram at a certain level and visualize these clusters using a scatter plot.
