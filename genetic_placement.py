import random
from math import floor, ceil
from statistics import mean
import numpy as np
import copy

from utils import *

class gene():
	def __init__(self, cell_num):
		self.cell_num = cell_num

	def update_location(self, x, y):
		self.x = x
		self.y = y

class GeneticPlacement():
	def __init__(self, benchmark, generation_num, inversion_rate, mutation_rate, w_x=1, w_y=2, crossover_type="order", population_size=24, crossover_rate=1.0, plot_enable=False):
		self.benchmark = benchmark
		self.ny, self.nx, self.total_cell_num, self.nets = preprocess("benchmarks_a2/"+benchmark+".txt")
		self.crossover_rate = crossover_rate
		self.inversion_rate = inversion_rate
		self.mutation_rate = mutation_rate
		self.w_x = w_x
		self.w_y = w_y
		self.crossover_type = crossover_type
		self.population_size = population_size
		self.generation_num = generation_num
		self.plot_enable = plot_enable

		self.historic_mean_cost = []
		self.historic_min_cost = []
		self.historic_diff = []

		random.seed(10)
		self.init_population()

	def run(self):
		print("**********"+self.benchmark+"**********")
		for i in range(self.generation_num):
			self.generation()
		max_f = 0
		solution = []
		for chromosome in self.population:
			f, _ = self.evaluate_fitness(chromosome)
			if f > max_f:
				max_f = f
				solution = chromosome
		self.plot(final_solution=True)
		print("Final cost:"+str(self.evaluate_fitness(solution)[1]))
		return solution

	# The operation for each generation
	def generation(self):
		assert(len(self.population)==self.population_size)
		fitness_of_population = []
		cost_of_population = []
		for chromosome in self.population:
			fitness, cost = self.evaluate_fitness(chromosome)
			fitness_of_population.append(fitness)
			cost_of_population.append(cost)
			if random.random() < self.inversion_rate:
				chromosome = self.inversion(chromosome)
		self.historic_mean_cost.append(mean(cost_of_population))
		self.historic_min_cost.append(min(cost_of_population))
		self.historic_diff.append(max(cost_of_population)-min(cost_of_population))
		if self.plot_enable:
			self.plot()
		f_prob = [float(i)/sum(fitness_of_population) for i in fitness_of_population]

		for chromosome in self.population:
			self.test_chromosome(chromosome)
		for i in range(floor(self.population_size*self.crossover_rate)):
			i1, i2 = np.random.choice(self.population_size, 2, replace=False, p=f_prob)
			p1, p2 = self.population[i1], self.population[i2]
			offspring = self.crossover(p1, p2)
			if random.random() < self.mutation_rate:
				offspring = self.mutation(offspring)
			self.population.append(offspring)

		for chromosome in self.population[self.population_size:]:
			fitness, _ = self.evaluate_fitness(chromosome)
			fitness_of_population.append(fitness)

		# ***** Random Selection ***** #
		# Random Selection does not work as well so we didn't go with this
		# selection = random.sample(range(floor(self.population_size*(1+self.crossover_rate))), self.population_size)
		# ***** Fitness Selection ***** #
		selection = np.argpartition(fitness_of_population, -self.population_size)[-self.population_size:]
		self.population = [self.population[i] for i in selection]

	def init_population(self):
		self.population = []
		for i in range(self.population_size):
			chromosome = []
			cells = list(range(self.total_cell_num))
			random.shuffle(cells)
			x = 0
			y = random.randint(0, self.ny-1-ceil(self.nx/self.total_cell_num))
			for cell_num in cells:
				new_gene = gene(cell_num)
				new_gene.update_location(x, y)
				chromosome.append(new_gene)
				x += 1
				if x >= self.nx:
					x = 0
					y += 1
				if y >= self.ny:
					y = 0
			self.population.append(chromosome)

	def evaluate_fitness(self, chromosome):
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
			assert(x_bound[1] >= x_bound[0] and y_bound[1] >= y_bound[0])
			assert(x_bound[0] < self.nx and y_bound[0] < self.ny)
			assert(x_bound[1] >= 0 and y_bound[1] >= 0)
			cost += self.w_x*(x_bound[1]-x_bound[0]) + self.w_y*(y_bound[1]-y_bound[0])
		return 1/cost, cost

	def test_chromosome(self, chromosome):
		cell = set()
		for gene in chromosome:
			assert(gene.cell_num not in cell)
			cell.add(gene.cell_num)

	def mutation(self, chromosome):
		i1, i2 = random.sample(range(self.total_cell_num), 2)
		temp = chromosome[i1].cell_num
		chromosome[i1].cell_num =  chromosome[i2].cell_num
		chromosome[i2].cell_num = temp
		return chromosome

	def inversion(self, chromosome):
		indices = random.sample(range(self.total_cell_num), 2)
		indices.sort()
		i1, i2 = indices
		while i1 < i2:
			temp = chromosome[i1]
			chromosome[i1] =  chromosome[i2]
			chromosome[i2] = temp
			i1 += 1
			i2 -= 1
		return chromosome

	def crossover(self, p1, p2):
		cut_i = random.randint(1, self.total_cell_num-2)
		offspring = [None]*self.total_cell_num
		if self.crossover_type == "order":
			offspring[:cut_i] = copy.deepcopy(p1[:cut_i])
			finalized_genes = set()
			occupied_locations = set()
			for g_o in offspring[:cut_i]:
				finalized_genes.add(g_o.cell_num)
				occupied_locations.add((g_o.x, g_o.y))
			i = cut_i
			for g_2 in p2:
				if g_2.cell_num not in finalized_genes:
					if (g_2.x, g_2.y) not in occupied_locations:
						offspring[i] = copy.deepcopy(g_2)
					else:
						found = -1
						for g_1 in p1[cut_i:]:
							if g_1.cell_num == g_2.cell_num:
								offspring[i] = copy.deepcopy(g_1)
								found = 1
						assert(found==1)
					i += 1
		# *****Implementation of PMT and Cycle Crossover, unused*****
		# elif self.crossover_type == "PMT":
		# 	offspring[cut_i:] = p2[cut_i:]
		# 	for i in range(cut_i,self.cell_num):
		# 		if p1[i].cell_num == p2[i].cell_num:
		# 			continue
		# 		for j in range(self.cell_num):
		# 			if p1[j].cell_num == p2[i].cell_num:
		# 				temp = p1[j]
		# 				p1[j] = p1[i]
		# 				p1[i] = p1[j]
		# 	offspring[:cut_i] = p1[:cut_i]
		# elif self.crossover_type == "cycle":
		# 	p_src = p1
		# 	p_ref = p2
		# 	while next_i != -1:
		# 		visited = set()
		# 		while True:
		# 			offspring[next_i] = p_src[next_i]
		# 			visited.add(p_src[next_i])
		# 			for i,gene in p_src:
		# 				if gene.cell_num == p_ref[next_i].cell_num:
		# 					next_i = i
		# 					break
		# 			if p_src[next_i] in visited:
		# 				break
		# 		if p_src is p1:
		# 			p_src = p2
		# 			p_ref = p1
		# 		else:
		# 			p_src = p1
		# 			p_ref = p2
		# 		next_i = -1
		# 		for i,gene in enumerate(offspring):
		# 			if gene is None:
		# 				next_i = i
		# 				break
		return offspring

	def plot(self, final_solution=False):
		plt.title('cost over generations')
		plt.plot(self.historic_mean_cost)

		x = np.arange(len(self.historic_mean_cost))
		plt.bar(x, self.historic_diff, bottom=self.historic_min_cost)

		plt.show(block=False)
		if final_solution:
			plt.savefig("./fig/placement"+self.benchmark+".png")
		plt.pause(0.5)
		plt.close()