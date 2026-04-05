


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

# Загрузка данных
iris = load_iris()
X = iris.data[:, :2]  # Берем первые 2 признака для 2D визуализации
y = iris.target

# Метод локтя (Elbow Method)
inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# График метода локтя
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Инерция (WCSS)')
plt.title('Метод локтя для определения оптимального k')
plt.grid(True, alpha=0.3)

# Silhouette Score
plt.subplot(1, 2, 2)
silhouette_scores = []
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

plt.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Количество кластеров (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score для определения оптимального k')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Вывод оптимального количества
optimal_k_elbow = 3  # По методу локтя
optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]

print(f"Оптимальное k по методу локтя: {optimal_k_elbow}")
print(f"Оптимальное k по Silhouette Score: {optimal_k_silhouette}")
print(f"Silhouette Score для k=3: {silhouette_scores[1]:.4f}")