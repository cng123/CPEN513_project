import random
import math
import numpy as np
from utils import preprocess

class gene():
	def __init__(cell_num, serial_num):
		self.cell_num = cell_num
		self.serial_num = serial_num

	def update_location(x,y):
		self.x = x
		self.y = y

class GeneticPlacement():
	def __init__(benchmark):
		self.ny, self.nx, self.total_cell_num, self.nets = preprocess("benchmarks/"+benchmark+".txt")
		self.grid = np.full((ny,nx), -1)

	def init_population(population_size):
		self.population = []
		for i in range(population_size):
			chromosome = []
			# randomly generate initial positions
			random.seed(10)
			rand_positions = random.sample(range(self.ny*self.nx), self.total_cell_num)
			# decode pos into x, y coordinates, and update on grid
			for cell_num, pos in enumerate(rand_positions):
				new_gene = gene(cell_num, cell_num)
				new_gene.update_location(floor(n/nx), n%nx)
				self.grid[floor(n/nx)][n%nx] = cell_num
				chromosome.append(new_gene)
			self.population.append(chromosome)

	def crossover():

	def mutation(chromosome):
		i1, i2 = random.sample(range(self.total_cell_num), 2)
		temp = chromosome[i1]
		chromosome[i1] =  chromosome[i2]
		chromosome[i2] = temp
		return chromosome

	def inversion(chromosome):
		indices = random.sample(range(10), 2)
		indices.sort()
		i1, i2 = indices
		while i1 < i2:
			temp = chromosome[i1]
			chromosome[i1] =  chromosome[i2]
			chromosome[i2] = temp
			i1 += 1
			i2 -= 1
		return chromosome

	def evaluate_fitness(chromosome):
		cost = 0
		for net in self.nets:
			x_bound = [self.nx,0]
			y_bound = [self.ny,0]
			for gene in chromosome:
				if gene.cell_num in net:
					if gene.x < x_bound[0]:
						x_bound[0] = gene.x
					if gene.x > x_bound[1]:
						x_bound[1] = gene.x
					if gene.y < y_bound[0]:
						y_bound[0] = gene.y
					if gene.y > y_bound[1]:
						y_bound[1] = gene.y
			cost += self.w_x*(x_bound[1]-x_bound[0]) + self.w_y*(y_bound[1]-y_bound[0])
		return 1/cost



