import numpy as np
import matplotlib.pyplot as plt

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

def rastrigin(x):
    return 10 * len(x) + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)

def rosenbrock(x):
    return sum(100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))

def schwefel(x):
    return 418.9829 * len(x) - sum(xi * np.sin(np.sqrt(abs(xi))) for xi in x)

def parabolic(x):
    return sum(xi**2 for xi in x)

def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(np.cos(c * xi) for xi in x)
    return -a * np.exp(-b * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + a + np.e

def griewank(x):
    sum1 = sum(xi**2 / 4000 for xi in x)
    prod = np.prod(np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x))
    return 1 + sum1 - prod

def michalewicz(x):
    m = 10
    return -sum(np.sin(xi) * (np.sin((i + 1) * xi**2 / np.pi))**(2 * m) for i, xi in enumerate(x))

def dixon_price(x):
    return (x[0] - 1)**2 + sum((i + 1) * (2 * x[i]**2 - x[i - 1])**2 for i in range(1, len(x)))

def sphere(x):
    return -sum(xi**2 for xi in x)

def sum_squares(x):
    return -sum((i + 1) * xi**2 for i, xi in enumerate(x))

def trid(x):
    return -(sum((xi - 1)**2 for xi in x) - sum(x[i] * x[i - 1] for i in range(1, len(x))))

def zakharov(x):
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(0.5 * (i + 1) * xi for i, xi in enumerate(x))
    return -(sum1 + sum2**2 + sum2**4)

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

# Адаптивный комбинированный поиск (максимизация)
def ultimate_hybrid_optimization(num_owls, num_iterations, fitness_counter, dimension=10, seed=42):
    """
    Универсальный гибридный метод оптимизации.
    """
    np.random.seed(seed)
    owls = np.random.uniform(-100, 100, size=(num_owls, dimension))  # Инициализация популяции
    best_fitness = -np.inf  # Максимизация
    best_owl = None
    fitness_history = []
    calls_history = []

    # Параметры адаптации
    gamma_0 = 1.0  # Начальный шаг глобального поиска
    gamma_min = 0.01  # Минимальный шаг
    elitism_rate = 0.1  # Процент элитных агентов
    migration_rate = 0.2  # Процент агентов, участвующих в миграции между популяциями
    local_search_rate = 0.1  # Вероятность применения локального поиска

    # Разделение популяции на группы (мультипопуляционный подход)
    num_groups = 3  # Число групп
    group_size = num_owls // num_groups
    populations = [owls[i * group_size:(i + 1) * group_size] for i in range(num_groups)]

    for iteration in range(num_iterations):
        new_populations = []
        for group_index, group in enumerate(populations):
            # Оценка значений фитнеса
            fitness = np.array([fitness_counter.evaluate(owl) for owl in group])
            best_index = fitness.argmax()  # Индекс лучшего агента в группе
            if fitness[best_index] > best_fitness:
                best_fitness = fitness[best_index]
                best_owl = group[best_index]

            # Сортируем агентов группы
            sorted_indices = np.argsort(fitness)[::-1]
            elite_count = max(1, int(elitism_rate * len(group)))  # Число элитных агентов
            elite_owls = group[sorted_indices[:elite_count]]  # Сохраняем лучших

            # Адаптивный шаг
            if iteration < num_iterations // 2:
                gamma = max(gamma_min, gamma_0 * (1 - iteration / num_iterations))
            else:
                gamma = gamma_min

            # Мутация для остальных агентов
            mutated_group = group + gamma * np.random.randn(len(group), dimension)
            mutated_group = np.clip(mutated_group, -100, 100)

            # Локальный поиск для части агентов
            for i in range(len(mutated_group)):
                if np.random.rand() < local_search_rate:
                    mutated_group[i] += 0.1 * np.sign(np.random.randn(dimension))  # Локальное улучшение

            # Обновляем популяцию группы
            mutated_group[:elite_count] = elite_owls  # Добавляем элитных агентов
            new_populations.append(mutated_group)

        # Миграция между группами
        for group_index, group in enumerate(new_populations):
            migrants = int(migration_rate * len(group))
            if group_index < len(new_populations) - 1:
                new_populations[group_index][:migrants] = new_populations[group_index + 1][:migrants]
            else:
                new_populations[group_index][:migrants] = new_populations[0][:migrants]

        populations = new_populations  # Обновляем популяции

        # Сохраняем историю
        fitness_history.append(best_fitness)
        calls_history.append(fitness_counter.calls)

    return best_owl, best_fitness, fitness_history, calls_history

def plot_convergence(fitness_histories, calls_histories, labels):
    plt.figure(figsize=(14, 6))

    # График сходимости по вызовам функции
    plt.subplot(1, 2, 2)
    for fitness_history, calls_history, label in zip(fitness_histories, calls_histories, labels):
        plt.plot(calls_history, fitness_history, label=label)
    plt.xlabel('Число вызовов функции')
    plt.ylabel('Лучшее значение фитнеса')
    plt.title('Сходимость по вызовам функции')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Запуск тестов
if __name__ == "__main__":
    num_owls = 50  # Размер популяции
    num_iterations = 1000  # Число итераций
    dimension = 10  # Размерность задачи

    # Тестирование разных функций
    test_functions = [rastrigin, rosenbrock, schwefel, parabolic, ackley, michalewicz, dixon_price, sphere, sum_squares, trid, zakharov]
    function_names = ['rastrigin', 'rosenbrock', 'schwefel', 'parabolic', 'ackley', 'michalewicz', 'dixon_price', 'sphere', 'sum_squares', 'trid', 'zakharov']

    for test_function, function_name in zip(test_functions, function_names):
        print(f"Тестирование {function_name}...")

        fitness_counter = FitnessCounter(test_function)

        # Запуск методов
        fitness_counter.reset()
        best_owl_sofa, best_fitness_sofa, fitness_history_sofa, calls_history_sofa = sofa_method(num_owls, num_iterations, fitness_counter, dimension)

        fitness_counter.reset()
        best_owl_aniso, best_fitness_aniso, fitness_history_aniso, calls_history_aniso = aniso_mutations_method(num_owls, num_iterations, fitness_counter, dimension)

        fitness_counter.reset()
        best_owl_combined, best_fitness_combined, fitness_history_combined, calls_history_combined = combined_search(num_owls, num_iterations, fitness_counter, dimension)

        fitness_counter.reset()
        best_owl_adaptive, best_fitness_adaptive, fitness_history_adaptive, calls_history_adaptive = ultimate_hybrid_optimization(num_owls, num_iterations, fitness_counter, dimension)

        # Визуализация всех методов для текущей функции
        plot_convergence(
            [fitness_history_sofa, fitness_history_aniso, fitness_history_combined, fitness_history_adaptive],
            [calls_history_sofa, calls_history_aniso, calls_history_combined, calls_history_adaptive],
            ['SOFA', 'Анизотропные мутации', 'Комбинированный поиск', 'Наша модификация']
        )
