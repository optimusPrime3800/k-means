import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from config import RESULTS_DIR, DPI

def load_iris_data():
    iris = load_iris()
    X = iris.data[:, :2]
    y = iris.target
    return X, y, iris.target_names

def find_optimal_k(X, k_range=range(2, 11)):
    inertias = []
    scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        scores.append(silhouette_score(X, labels))
    
    optimal_k = k_range[np.argmax(scores)]
    return optimal_k, inertias, scores, k_range

def plot_optimal_k(inertias, scores, k_range):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2)
    axes[0].set_xlabel('K'); axes[0].set_ylabel('Инерция')
    axes[0].set_title('Метод локтя'); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(k_range, scores, 'ro-', linewidth=2)
    axes[1].set_xlabel('K'); axes[1].set_ylabel('Silhouette')
    axes[1].set_title('Silhouette Score'); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/01_optimal_k.png', dpi=DPI)
    plt.show()

def calculate_metrics(y_true, labels):
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)
    return ari, nmi