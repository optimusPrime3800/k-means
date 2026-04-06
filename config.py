import os

# Пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Параметры алгоритма
N_CLUSTERS = 3
MAX_ITERATIONS = 50
RANDOM_STATE = 42

# Визуализация
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1']
FIG_SIZE = (10, 8)
DPI = 150

# Оси графика (для ирисов)
X_LIMITS = (4, 8)
Y_LIMITS = (1.5, 4.5)

def ensure_results_dir():
    """Создаёт папку для результатов, если нет"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    return RESULTS_DIR