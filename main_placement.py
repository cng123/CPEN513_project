import numpy as np

from genetic_placement import *
from utils import print_chromosome

if __name__ == "__main__":
	crossover_rate = 0
	crossover_rate = 0
	mutation_rate = 0
	genetic_placer = GeneticPlacement("cm138a", crossover_rate, crossover_rate, mutation_rate, 1, 2)
	genetic_placer.init_population(2)

	# print_chromosome(genetic_placer.population[0])

	# genetic_placer.inversion(genetic_placer.population[0])

	# print_chromosome(genetic_placer.population[0])

	# genetic_placer.mutation(genetic_placer.population[0])

	# print_chromosome(genetic_placer.population[0])
	# print_chromosome(genetic_placer.population[1])

	# # genetic_placer.inversion(genetic_placer.population[0])

	# # print_chromosome(genetic_placer.population[0])

	# offspring = genetic_placer.crossover(genetic_placer.population[0],genetic_placer.population[1])
	# print_chromosome(offspring)

	# offspring = genetic_placer.crossover(genetic_placer.population[1],genetic_placer.population[0])
	# print_chromosome(offspring)

	# offspring = genetic_placer.crossover(genetic_placer.population[0],genetic_placer.population[1])
	# print_chromosome(offspring)

	# print(genetic_placer.evaluate_fitness(genetic_placer.population[0]))
	# print(genetic_placer.evaluate_fitness(offspring))