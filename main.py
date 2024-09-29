import math
from functools import cmp_to_key
from random import randint, shuffle, uniform
from time import sleep
import matplotlib.pyplot as plt
from data_test import cities

SIZE_X = 200
SIZE_Y = 200
INDIVIDUAL_SIZE = 20
POPULATION_SIZE = 80
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
CITIES_NUMBER = 20


def generate_city(max_x, max_y):
    x = randint(1, max_x)
    y = randint(1, max_y)
    return {"x": x, "y": y}


def generate_city_list():
    # cities = []
    # for i in range(CITIES_NUMBER):
    #     cities.append(generate_city(SIZE_X, SIZE_Y))
    return cities


def generate_population(cities, length):
    population = []
    for _ in range(length):
        cities_copy = cities[:]
        shuffle(cities_copy)
        population.append(cities_copy)
    return population


def calculate_path(city1, city2):
    len_x = abs(city1["x"] - city2["x"])
    len_y = abs(city1["y"] - city2["y"])
    path = math.sqrt(len_x ** 2 + len_y ** 2)
    return path


def fitness_function(individual):
    fitness = 0
    for gen1 in range(len(individual) - 1):
        gen2 = gen1 + 1
        fitness += calculate_path(individual[gen1], individual[gen2])
    return fitness


def crossover(ind1, ind2):
    # decide if we will make a crossover
    if uniform(0, 1) > CROSSOVER_RATE: return ind1, ind2
    crossover_point1 = randint(1, INDIVIDUAL_SIZE - 1)
    crossover_point2 = randint(1, INDIVIDUAL_SIZE - 1)
    if crossover_point1 > crossover_point2:
        crossover_point2, crossover_point1 = crossover_point1, crossover_point2
    ind1_start, ind1_end = ind1[:crossover_point1], ind1[crossover_point2:]
    ind2_start, ind2_end = ind2[:crossover_point1], ind2[crossover_point2:]
    new_ind1 = ind1_start
    new_ind1 += [city for city in ind2 if (city not in ind1_start) and (city not in ind1_end)]
    new_ind1 += ind1_end
    new_ind2 = ind2_start
    new_ind2 += [city for city in ind1 if (city not in ind2_start) and (city not in ind2_end)]
    new_ind2 += ind2_end
    return new_ind1, new_ind2


def mutate(ind):
    if uniform(0, 1) > MUTATION_RATE: return ind
    mutate_index1 = randint(0, INDIVIDUAL_SIZE-1)
    mutate_index2 = mutate_index1
    while mutate_index1 == mutate_index2:
        mutate_index2 = randint(0, INDIVIDUAL_SIZE-1)
    individual = ind[:]
    individual[mutate_index1], individual[mutate_index2] = individual[mutate_index2], individual[mutate_index1]
    return ind


def roulette_selection(wheel):
    index = uniform(0, 1)
    for i in range(len(wheel)):
        if index <= wheel[i][1]:
            selection = wheel[i][0]
            return selection


def sum_fitness(individuals):
    summa = 0
    for fitness in individuals:
        summa += fitness[1]
    return summa


# add the remaining elements using crossover and mutation
def fill_rest(population, number_of_individuals):
    new_rest_population = []
    fitness_total = sum_fitness(population)
    # calculate wheel index for selection
    # each part of wheel has its own index
    # that is calculated: probability of current part + sum of all previous indexes
    wheel = []
    prev_index = 0
    for p in population:
        probability = p[1] / fitness_total
        current_index = probability + prev_index
        prev_index = current_index
        wheel.append((p[0], current_index))

    for _ in range(math.floor(number_of_individuals / 2)):
        individual1 = roulette_selection(wheel)
        individual2 = individual1
        while individual1 == individual2:
            individual2 = roulette_selection(wheel)
        individual1, individual2 = crossover(individual1, individual2)
        individual1 = mutate(individual1)
        individual2 = mutate(individual2)
        new_rest_population.append(individual1)
        new_rest_population.append(individual2)
    return new_rest_population


def fitness_compare(item1, item2):
    return item1[1] - item2[1]


def next_generation(population):
    # list contains (individual, individual fitness)
    population_fitness = []
    new_population = []
    for i in range(len(population)):
        fitness = fitness_function(population[i])
        population_fitness.append((population[i], fitness))
    # print("\n Printing fitness")
    # for f in population_fitness:
    #     print(f)

    # best selection
    # select the best one individual and 10% of the rest best
    rest_best = math.floor(INDIVIDUAL_SIZE * 0.2) + 1
    # the rest number of individuals must be even
    if (POPULATION_SIZE - (rest_best + 1)) % 2 != 0: rest_best += 1
    rest_number = POPULATION_SIZE - rest_best
    sorted_fitness = sorted(population_fitness, key=cmp_to_key(fitness_compare))
    best_path = sorted_fitness[0]
    unique_fitness = []
    index = 0
    individual = best_path
    for _ in range(rest_best):
        while individual[1] in unique_fitness:
            index += 1
            individual = sorted_fitness[index]
        unique_fitness.append(individual[1])
        new_population.append(individual[0])

    new_population += fill_rest(population_fitness, rest_number)
    return new_population, best_path


def draw_path(generation, ax, path, path_length):
    ax.clear()
    x_coords = []
    y_coords = []
    for city in path:
        x_coords.append(city['x'])
        y_coords.append(city['y'])

    for i in range(len(x_coords)):
        plt.text(x_coords[i], y_coords[i], str(i+1), fontsize=12, ha='center', va='bottom')

    ax.plot(x_coords, y_coords, '--go', label='Best Route', linewidth=2.5)
    ax.set_title(f"Generation {generation}", fontsize=16)
    ax.text(0.5, 1.05, f"Path: {path_length}", fontsize=12, ha='center', transform=ax.transAxes)
    ax.set_xlim(0, SIZE_X)
    ax.set_ylim(0, SIZE_Y)
    ax.legend()
    plt.draw()
    plt.pause(0.1)


def start():
    cities = generate_city_list()
    population = generate_population(cities, POPULATION_SIZE)
    same_result_count = 0
    best_path = 0
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.ion()
    plt.show()
    generation = 1

    while True:
        population, new_best_path = next_generation(population)

        # if result doesn't change break the program
        if best_path == new_best_path[1]: same_result_count += 1
        else: same_result_count = 0
        if same_result_count == 150: break
        best_path = new_best_path[1]
        print(best_path)
        draw_path(generation, ax, new_best_path[0], new_best_path[1])
        generation += 1

        sleep(0.1)

    plt.show(block=True)


start()
