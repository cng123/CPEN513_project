import numpy as np

from genetic_placement import *
from utils import print_chromosome

if __name__ == "__main__":
	#The mutation rate can vary from 0 to 10% in steps of 0.5%
	# the inversion rate can vary from 0 to 100% in steps of 5%
	# the crossover rate can vary from 20% to l00% in steps of 4%
	inversion_rate = 0.5
	mutation_rate = 0.05
	genetic_placer = GeneticPlacement("cm138a", 100, inversion_rate, mutation_rate, 1, 2)
	genetic_placer.run()