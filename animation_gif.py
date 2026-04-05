from matplotlib.animation import FuncAnimation, PillowWriter

fig, ax = plt.subplots(figsize=(10, 8))

def init():
    ax.clear()
    ax.set_xlim(4, 8)
    ax.set_ylim(1.5, 4.5)
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Sepal Width (cm)')
    ax.set_title('K-Means Clustering Process')
    ax.grid(True, alpha=0.3)
    return []

def animate(i):
    ax.clear()
    hist = kmeans.history[i]
    
    # Рисуем точки
    for cluster in range(3):
        mask = hist['labels'] == cluster
        ax.scatter(X[mask, 0], X[mask, 1], 
                  c=colors[cluster], alpha=0.6, 
                  s=50, edgecolors='black', linewidth=0.5)
    
    # Рисуем центроиды
    ax.scatter(hist['centroids'][:, 0], hist['centroids'][:, 1],
              c='black', s=300, marker='X', 
              edgecolors='white', linewidth=2)
    
    # Легенда
    patches = [mpatches.Patch(color=colors[j], label=f'Cluster {j}') 
               for j in range(3)]
    patches.append(mpatches.Patch(color='black', label='Centroid'))
    ax.legend(handles=patches, loc='upper right')
    
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Sepal Width (cm)')
    ax.set_title(f'Итерация {i + 1} / {len(kmeans.history)}')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(4, 8)
    ax.set_ylim(1.5, 4.5)
    
    return []

anim = FuncAnimation(fig, animate, init_func=init,
                    frames=len(kmeans.history),
                    interval=800, blit=False)

# Сохранение как GIF
anim.save('kmeans_animation.gif', writer=PillowWriter(fps=2))
plt.show()

print(f"Анимация сохранена: {len(kmeans.history)} кадров")