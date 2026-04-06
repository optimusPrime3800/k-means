import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.patches as mpatches
from config import COLORS, FIG_SIZE, X_LIMITS, Y_LIMITS, DPI, RESULTS_DIR

def plot_iterations(X, history, n_plots=6):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, hist in enumerate(history[:n_plots]):
        ax = axes[i]
        for cluster in range(len(COLORS)):
            mask = hist['labels'] == cluster
            ax.scatter(X[mask, 0], X[mask, 1], c=COLORS[cluster], alpha=0.6, s=50, edgecolors='black')
        
        ax.scatter(hist['centroids'][:, 0], hist['centroids'][:, 1], c='black', s=300, marker='X', edgecolors='white')
        ax.set_title(f'Итерация {hist["iteration"] + 1}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(X_LIMITS); ax.set_ylim(Y_LIMITS)
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/02_iterations.png', dpi=DPI, bbox_inches='tight')
    plt.show()

def create_animation(X, history):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    def init():
        ax.set_xlim(X_LIMITS); ax.set_ylim(Y_LIMITS)
        ax.set_xlabel('Sepal Length'); ax.set_ylabel('Sepal Width')
        ax.set_title('K-Means Process'); ax.grid(True, alpha=0.3)
        return []
    
    def animate(i):
        ax.clear()
        hist = history[i]
        for cluster in range(len(COLORS)):
            mask = hist['labels'] == cluster
            ax.scatter(X[mask, 0], X[mask, 1], c=COLORS[cluster], alpha=0.6, s=50, edgecolors='black')
        ax.scatter(hist['centroids'][:, 0], hist['centroids'][:, 1], c='black', s=300, marker='X', edgecolors='white')
        ax.set_title(f'Итерация {i + 1} / {len(history)}')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(X_LIMITS); ax.set_ylim(Y_LIMITS)
        return []
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=len(history), interval=800, blit=False)
    anim.save(f'{RESULTS_DIR}/03_animation.gif', writer=PillowWriter(fps=2))
    plt.close()
    print("✓ Анимация сохранена")

def plot_final_comparison(X, y_true, labels, centroids, ari, nmi):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Ground Truth
    for i, color in enumerate(COLORS):
        mask = y_true == i
        axes[0].scatter(X[mask, 0], X[mask, 1], c=color, alpha=0.6, s=50, edgecolors='black', label=f'Class {i}')
    axes[0].set_title('Истинные метки'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(X_LIMITS); axes[0].set_ylim(Y_LIMITS)
    
    # K-Means
    for i, color in enumerate(COLORS):
        mask = labels == i
        axes[1].scatter(X[mask, 0], X[mask, 1], c=color, alpha=0.6, s=50, edgecolors='black')
    axes[1].scatter(centroids[:, 0], centroids[:, 1], c='black', s=300, marker='X', edgecolors='white')
    axes[1].set_title(f'K-Means Результат'); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(X_LIMITS); axes[1].set_ylim(Y_LIMITS)
    
    # Metrics
    axes[2].axis('off')
    text = f"Метрики:\n\nARI: {ari:.4f}\nNMI: {nmi:.4f}\n\nКластеров: {len(centroids)}"
    axes[2].text(0.1, 0.5, text, fontsize=14, family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/04_final.png', dpi=DPI, bbox_inches='tight')
    plt.show()