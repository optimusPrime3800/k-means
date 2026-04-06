

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data[:, :2]  # Первые 2 признака для 2D

y_true = iris.target


class KMeansCustom:

    def __init__(self, n_clusters, max_iters=100, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.history = []  # Для хранения истории итераций
        
        
    def assign_clusters(self, X, centroids):
        distances = np.zeros((len(X), self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.sqrt(np.sum((X - centroids[i])**2, axis=1))
        return np.argmin(distances, axis=1)



    def update_centroids(self, X, labels):
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for i in range(self.n_clusters):
            mask = labels = i
            if np.sum(mask) > 0:
                new_centroids[i] = X[mask].mean(axis=0)
            else:
                new_centroids[i] = self.centroids[i]
        return new_centroids        



    def create_centroids(self, X):
        rng = np.random.default_rng(self.random_state)
        indices = np.random.choice(len(X), self.n_clusters, replace=False)
        return X[indices].copy()

        

    def fit(self, X):
        self.centroids = self.create_centroids(X)
        self.history = []
            
        for iteration in range(self.max_iters):
            # сохраняем состояние перед итерацией
            old_centroids = self.centroids.copy()
            old_labels =  self.labels.copy() if self.labels is not None else None
            self.labels = self.assign_clusters(X, self.centroids)
            self.history.append({
                    'iteration': iteration,
                    'centroids': self.centroids.copy(),
                    'labels': self.labels.copy(),
                    'old_centroids': old_centroids,
                    'old_labels': old_labels
                })
            self.centroids = self.update_centroids(X, self.labels)

            if np.allclose(old_centroids, self.centroids):
                print(f"сходимость достигнута на итерации {iteration + 1}")
                break

        print(f"всего итераций: {len(self.history)}")
        return self


kmeans = KMeansCustom(n_clusters=3, max_iters=50)

kmeans.fit(X)



colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
color_names = ['Cluster 0', 'Cluster 1', 'Cluster 2']