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
	def __init__(benchmark, crossover_rate, inversion_rate, mutation_rate):
		self.ny, self.nx, self.total_cell_num, self.nets = preprocess("benchmarks/"+benchmark+".txt")
		self.crossover_rate = crossover_rate
		self.inversion_rate = inversion_rate
		self.mutation_rate = mutation_rate
		#The mutation rate as given by this parameter can vary from 0 to 10% in steps of
		# 0.5%. the inversion rate can vary from 0 to 100% in steps of 5%. and the crossover rate can vary from 20% to loo%,
		# in steps of 4%

	def init_population(population_size):
		self.population = []
		for i in range(population_size):
			chromosome = []
			# randomly generate initial positions
			random.seed(10)
			rand_positions = random.sample(range(self.ny*self.nx), self.total_cell_num)
			# decode pos into x, y coordinates, and update on grid
			for cell_num, pos in enumerate(rand_positions):
				new_gene = gene(cell_num, pos)
				new_gene.update_location(floor(n/nx), n%nx)
				self.grid[floor(n/nx)][n%nx] = cell_num
				chromosome.append(new_gene)
			self.population.append(chromosome)

	def crossover(p1, p2):
		cut_i = random.randint(0, self.total_cell_num-1)
		offspring = [None]*self.total_cell_num
		if self.crossover_type == "order":
			offspring[:cut_i] = p1[:cut_i]
			finalized_genes = set()
			for g_o in offspring[:cut_i]:
				finalized_genes.add(g_o.cell_num)
			i = cut_i+1
			for g_2 in p2:
				if g_2.cell_num not in finalized_genes:
					offspring[:i] = g_2
					i += 1
		elif self.crossover_type == "PMT":
			offspring[cut_i:] = p2[cut_i:]
			for i in range(cut_i,self.cell_num):
				if p1[i].cell_num == p2[i].cell_num:
					continue
				for j in range(self.cell_num):
					if p1[j].cell_num == p2[i].cell_num:
						temp = p1[j]
						p1[j] = p1[i]
						p1[i] = p1[j]
			offspring[:cut_i] = p1[:cut_i]
		elif self.crossover_type == "cycle":
			p_src = p1
			p_ref = p2
			while None in offspring:
				visited = set()
				next_i = 0
				for i,gene in enumerate(offspring):
					if gene is None:
						next_i = i
						break
				while True:
					offspring[next_i] = p_src[next_i]
					visited.add(p_src[next_i])
					for i,gene in p_src:
						if gene.cell_num == p_ref[next_i].cell_num:
							next_i = i
							break
					if p_src[next_i] in visited:
						break


				if p_src is p1:
					p_src = p2
					p_ref = p1
				else:
					p_src = p1
					p_ref = p2


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



