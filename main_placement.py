import numpy as np

from genetic_placement import *
from utils import print_chromosome

benchmarks = ["alu2","apex1","apex4","c880","cm138a","cm150a","cm151a","cm162a","cps","e64","paira","pairb"]
# benchmarks = ["benchmark_name"]

if __name__ == "__main__":
	#The mutation rate can vary from 0 to 10% in steps of 0.5%
	# the inversion rate can vary from 0 to 100% in steps of 5%
	# the crossover rate can vary from 20% to l00% in steps of 4%
	crossover_rate = 1.0
	inversion_rate = 0.5
	mutation_rate = 0.05
	generation_num = 100
	w_x = 1
	w_y = 2
	population_size = 24
	for benchmark in benchmarks:
		genetic_placer = GeneticPlacement(benchmark, generation_num, inversion_rate, mutation_rate, 
				population_size=population_size, w_x=w_x, w_y=w_y, crossover_rate=crossover_rate, plot_enable=False)
		genetic_placer.run()