from config import ensure_results_dir, N_CLUSTERS
from analyzer import load_iris_data, find_optimal_k, plot_optimal_k, calculate_metrics
from kmeans import KMeansCustom
from visualizer import plot_iterations, create_animation, plot_final_comparison

def main():
    print("=" * 50)
    print("🚀 Запуск K-Means проекта")
    print("=" * 50)
    
    # 1. Подготовка
    results_dir = ensure_results_dir()
    print(f"📁 Результаты: {results_dir}")
    
    # 2. Данные
    X, y_true, target_names = load_iris_data()
    print(f"📊 Данные загружены: {X.shape}")
    
    # 3. Подбор K
    print("\n🔍 Подбор оптимального K...")
    optimal_k, inertias, scores, k_range = find_optimal_k(X)
    plot_optimal_k(inertias, scores, k_range)
    print(f"✓ Оптимальное K: {optimal_k}")
    
    # 4. K-Means
    print(f"\n🤖 Запуск K-Means (k={N_CLUSTERS})...")
    kmeans = KMeansCustom(n_clusters=N_CLUSTERS, max_iters=50, random_state=42)
    kmeans.fit(X)
    
    # 5. Визуализация
    print("\n🎨 Визуализация...")
    plot_iterations(X, kmeans.history)
    create_animation(X, kmeans.history)
    
    # 6. Метрики
    ari, nmi = calculate_metrics(y_true, kmeans.labels)
    plot_final_comparison(X, y_true, kmeans.labels, kmeans.centroids, ari, nmi)
    
    # 7. Итог
    print("\n" + "=" * 50)
    print("✅ ГОТОВО!")
    print(f"ARI: {ari:.4f} | NMI: {nmi:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()