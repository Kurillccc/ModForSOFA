import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.integrate import trapezoid

# Параметры системы
params = {
    'kb': 1.0,
    'a0': 1.0,
    'a1': 1.0,
    'a2': 1.0,
    'a3': 1.0,
    'a4': 1.0,
    'a5': 1.0,
    'a6': 1.0,
    'k': 1.0,
    'xi': 1.0,
    'L': 1.0,
    'omega': 1.0,
    'y_star': 1.0
}

# Функция описывает динамику системы (Y - переменные)
def system(t, Y):
    Y1, Y2, Y3, Y4, Y5 = Y
    dY1_dt = -params['kb'] * Y1 + params['a0'] * Y2 + Y3 + params['k'] * params['xi'] * Y4
    dY2_dt = params['a1'] * Y1 - params['a2'] * Y2
    dY3_dt = params['a3'] * Y2 - params['a4'] * Y3
    dY4_dt = params['a5'] * Y3 - params['a6'] * Y4
    dY5_dt = params['L'] * (Y1 - params['y_star']) + params['omega'] * Y5
    return [dY1_dt, dY2_dt, dY3_dt, dY4_dt, dY5_dt]

# Функция для оценки фитнеса (значения интеграла)
def comp_int(owl, tk):
    # Начальные условия из текущей позиции "сова"
    Y0 = owl
    # Решение дифференциального уравнения
    sol = solve_ivp(system, [0, tk], Y0, t_eval=np.linspace(0, tk, 100))
    # Интегральное значение фитнеса (например, интеграл суммы квадратов переменных)
    integral = trapezoid(np.sum(sol.y**2, axis=0), sol.t)
    return integral


# Функция для подсчёта вызовов функции (FitnessCounter)
class FitnessCounter:
    def __init__(self, function):
        self.function = function
        self.calls = 0

    def evaluate(self, x):
        self.calls += 1
        return self.function(x)

    def reset(self):
        self.calls = 0

# SOFA метод (симуляция)
# SOFA метод (максимизация)
def sofa_method(num_owls, num_iterations, fitness_counter, dimension=10, seed=42):
    np.random.seed(seed)
    owls = np.random.uniform(-100, 100, size=(num_owls, dimension))
    best_fitness = -np.inf  # Для максимизации
    best_owl = None
    fitness_history = []
    calls_history = []

    for iteration in range(num_iterations):
        fitness = np.array([fitness_counter.evaluate(owl) for owl in owls])
        best_index = fitness.argmax()  # Индекс агента с максимальным значением
        if fitness[best_index] > best_fitness:
            best_fitness = fitness[best_index]
            best_owl = owls[best_index]

        owls += np.random.randn(num_owls, dimension)  # Агрессивная мутация
        owls = np.clip(owls, -100, 100)

        fitness_history.append(best_fitness)
        calls_history.append(fitness_counter.calls)

    return best_owl, best_fitness, fitness_history, calls_history

# Анизотропные мутации (максимизация)
def aniso_mutations_method(num_owls, num_iterations, fitness_counter, dimension=10, seed=42):
    np.random.seed(seed)
    owls = np.random.uniform(-100, 100, size=(num_owls, dimension))
    best_fitness = -np.inf  # Для максимизации
    best_owl = None
    fitness_history = []
    calls_history = []

    for iteration in range(num_iterations):
        fitness = np.array([fitness_counter.evaluate(owl) for owl in owls])
        best_index = fitness.argmax()  # Индекс агента с максимальным значением
        if fitness[best_index] > best_fitness:
            best_fitness = fitness[best_index]
            best_owl = owls[best_index]

        # Анизотропная мутация (величина мутации зависит от направления)
        owls += np.random.randn(num_owls, dimension) * (iteration / num_iterations)
        owls = np.clip(owls, -100, 100)

        fitness_history.append(best_fitness)
        calls_history.append(fitness_counter.calls)

    return best_owl, best_fitness, fitness_history, calls_history


# Комбинированный поиск (максимизация)
def combined_search(num_owls, num_iterations, fitness_counter, dimension=10, seed=42):
    np.random.seed(seed)
    owls = np.random.uniform(-100, 100, size=(num_owls, dimension))
    best_fitness = -np.inf  # Для максимизации
    best_owl = None
    fitness_history = []
    calls_history = []

    for iteration in range(num_iterations):
        fitness = np.array([fitness_counter.evaluate(owl) for owl in owls])
        best_index = fitness.argmax()  # Индекс агента с максимальным значением
        if fitness[best_index] > best_fitness:
            best_fitness = fitness[best_index]
            best_owl = owls[best_index]

        # Комбинированный поиск: мутация и локальное улучшение
        owls += np.random.randn(num_owls, dimension) * (0.5 + iteration / num_iterations)
        owls = np.clip(owls, -100, 100)

        fitness_history.append(best_fitness)
        calls_history.append(fitness_counter.calls)
    return best_owl, best_fitness, fitness_history, calls_history

# Модификация "Наша модификация"
# СОФА с динамическим управлением ресурсами
def dynamic_search(num_owls, num_iterations, fitness_counter, dimension, params_init, seed=42):
    np.random.seed(seed)

    # Инициализация сов
    owls = np.random.uniform(-100, 100, size=(num_owls, dimension))  # Диапазон параметров [-100, 100]
    best_fitness = -np.inf  # Для максимизации
    best_params = owls[0]
    fitness_progress = []

    global_mutation_scale = 1.0  # Увеличен масштаб мутаций
    local_step_scale = 0.1
    progress_threshold = 0.01
    min_global_fraction = 0.1
    global_fraction = 0.7  # Начальная доля глобального поиска

    recent_improvements = []

    for iteration in range(num_iterations):
        # Вычисляем фитнес для каждой "совы"
        fitness = np.array([fitness_counter.evaluate(owl) for owl in owls])

        # Обновляем лучшее решение
        best_index = fitness.argmax()
        if fitness[best_index] > best_fitness:
            best_fitness = fitness[best_index]
            best_params = owls[best_index]

        # Обновляем метрику прогресса
        if len(fitness_progress) > 1:
            improvement = fitness_progress[-1] - fitness_progress[-2]
            recent_improvements.append(improvement)

        # Рассчитываем средний прогресс за последние итерации
        if len(recent_improvements) >= 5:
            avg_progress = np.mean(recent_improvements[-5:])
            if avg_progress < progress_threshold:  # Низкий прогресс -> больше глобального поиска
                global_fraction = min(global_fraction + 0.1, 1.0)
            else:  # Высокий прогресс -> больше локального поиска
                global_fraction = max(global_fraction - 0.1, min_global_fraction)

        # Глобальный или локальный поиск
        for i in range(num_owls):
            if np.random.rand() < global_fraction:
                # Глобальный поиск
                owls[i] += np.random.normal(0, global_mutation_scale, size=dimension)
            else:
                # Локальный поиск
                grad = np.random.uniform(-0.5, 0.5, size=dimension)  # Случайный градиент
                owls[i] += local_step_scale * grad

        # Ограничиваем значения сов в пределах диапазона
        owls = np.clip(owls, -100, 100)

        # Сохраняем прогресс
        fitness_progress.append(best_fitness)

    return best_params, best_fitness, fitness_progress


# Визуализация сходимости
def plot_convergence(fitness_histories, labels):
    plt.figure(figsize=(8, 6))

    # График сходимости по итерациям
    plt.subplot()
    for history, label in zip(fitness_histories, labels):
        plt.plot(history, label=label)
    plt.xlabel('Итерации')
    plt.ylabel('Лучшее значение фитнеса')
    plt.title('Сходимость по итерациям')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# Пример целевой функции (Rastrigin)
def rastrigin(x):
    return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

def sphere(x):
    """
    Функция сферы (простая квадратичная функция):
    f(x) = sum(xi^2) для всех i
    """
    return sum(xi**2 for xi in x)

def ackley(x):
    """
    Функция Аклли — имеет несколько локальных минимумов:
    f(x) = -20 * exp(-0.2 * sqrt(1/n * sum(xi^2))) - exp(1/n * sum(cos(2*pi*xi))) + e + 20
    """
    n = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(np.cos(2 * np.pi * xi) for xi in x)
    term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    return term1 + term2 + 20 + np.e

def griewank(x):
    """
    Функция Гриванка — имеет много локальных минимумов:
    f(x) = 1 + (sum(xi^2) / 4000) - product(cos(xi / sqrt(i)))
    """
    sum1 = sum(xi**2 for xi in x) / 4000
    prod = np.prod([np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)])
    return 1 + sum1 - prod

def griewank(x):
    return 0
# -----------------------------------

def start(func):
    # Целевая функция
    test_function = func
    fitness_counter = FitnessCounter(test_function)

    # Запуск методов
    fitness_counter.reset()
    best_owl_sofa, best_fitness_sofa, fitness_history_sofa, calls_history_sofa = sofa_method(num_owls, num_iterations,fitness_counter, dimension)
    print("Метод СОФА: лучшее решение:",np.array2string(best_owl_sofa, separator=', ', precision=4, floatmode='fixed'),"Значение интеграла:", f"{best_fitness_sofa:.4f}")

    fitness_counter.reset()
    best_owl_aniso, best_fitness_aniso, fitness_history_aniso, calls_history_aniso = aniso_mutations_method(num_owls,num_iterations,fitness_counter,dimension)
    print("Метод СОФА с анизотропными мутациями: лучшее решение:",np.array2string(best_owl_aniso, separator=', ', precision=4, floatmode='fixed'),"Значение интеграла:", f"{best_fitness_aniso:.4f}")

    fitness_counter.reset()
    best_owl_combined, best_fitness_combined, fitness_history_combined, calls_history_combined = combined_search(num_owls, num_iterations, fitness_counter, dimension)
    print("Метод СОФА с комбинированным поиском: лучшее решение:",np.array2string(best_owl_combined, separator=', ', precision=4, floatmode='fixed'),"Значение интеграла:", f"{best_fitness_combined:.4f}")

    fitness_counter.reset()
    best_owl_adaptive, best_fitness_adaptive, fitness_history_adaptive, calls_history_adaptive = dynamic_search(num_owls, num_iterations, fitness_counter, dimension,params)
    print(f"\nМетод СОФА с динамическим управлением ресурсами: лучшее решение:: {best_owl_adaptive}")
    print(f"Значение интеграла:: {best_fitness_adaptive}")

    # Визуализация всех методов
    plot_convergence
    (
        [fitness_history_sofa, fitness_history_aniso, fitness_history_combined, fitness_history_adaptive],
        ['SOFA', 'Анизотропные мутации', 'Комбинированный поиск', 'Адаптивный комбинированный поиск']
    )

# Запуск тестов
if __name__ == "__main__":
    # params
    num_owls = 50  # Размер популяции
    num_iterations = 100  # Число итераций
    dimension = 10  # Размерность задачи

    # Start

    # Тестирование разных функций
    test_functions = [rastrigin, sphere, ackley, griewank]
    function_names = ['Rastrigin', 'Sphere', 'Ackley', 'Griewank']

    for test_function, function_name in zip(test_functions, function_names):
        print(f"Тестирование {function_name}...")

        fitness_counter = FitnessCounter(test_function)

        # Запуск методов
        fitness_counter.reset()
        best_owl_sofa, best_fitness_sofa, fitness_history_sofa, calls_history_sofa = sofa_method(num_owls,num_iterations,fitness_counter,dimension)
        print("Метод СОФА: лучшее решение:",np.array2string(best_owl_sofa, separator=', ', precision=4, floatmode='fixed'), "Значение интеграла:",f"{best_fitness_sofa:.4f}")

        fitness_counter.reset()
        best_owl_aniso, best_fitness_aniso, fitness_history_aniso, calls_history_aniso = aniso_mutations_method(num_owls, num_iterations, fitness_counter, dimension)
        print("Метод СОФА с анизотропными мутациями: лучшее решение:",np.array2string(best_owl_aniso, separator=', ', precision=4, floatmode='fixed'), "Значение интеграла:",f"{best_fitness_aniso:.4f}")

        fitness_counter.reset()
        best_owl_combined, best_fitness_combined, fitness_history_combined, calls_history_combined = combined_search(num_owls, num_iterations, fitness_counter, dimension)
        print("Метод СОФА с комбинированным поиском: лучшее решение:",np.array2string(best_owl_combined, separator=', ', precision=4, floatmode='fixed'), "Значение интеграла:",f"{best_fitness_combined:.4f}")

        fitness_counter.reset()
        best_owl_adaptive, best_fitness_adaptive, fitness_history_adaptive = dynamic_search(num_owls, num_iterations, fitness_counter, dimension, params)
        print(f"\nМетод СОФА с динамическим управлением ресурсами: лучшее решение:: {best_owl_adaptive}")
        print(f"Значение интеграла:: {best_fitness_adaptive}")

        # Визуализация всех методов для текущей функции
        plot_convergence(
            [fitness_history_sofa, fitness_history_aniso, fitness_history_combined, fitness_history_adaptive],
            ['SOFA', 'Анизотропные мутации', 'Комбинированный поиск', 'Наша модификация']
        )