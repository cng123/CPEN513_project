import random
import math
import time
import numpy as np
import itertools
from functools import reduce
import time

class GeneticPartition():
	def __init__(self, params, pop_count, hyper):
		random.seed(0)
		np.random.seed(0)
		self.num_cells = params["num_cells"]

		self.nets      = params["nets"]
		self.num_nets  = len(params["nets"])

		self.pop_count = pop_count

		# two ways to treat nets:
		# either treat each net as a hyper-edge, or treat each net as a clique
		self.hyper     = (hyper == "hyper") 

		if self.hyper:
			self.gain       = self.gain_hyper
			self.delta      = self.delta_hyper
			self.cost_verif = self.calc_cost
		else:
			self.gain       = self.gain_clique
			self.delta      = self.delta_clique
			self.cost_verif = self.calc_cost_clique

		# according to paper, 5 gives best results
		self.splits    = 5

		# according to paper, this is set to n/(3k) - 1, 
		# where n = number of nodes, and k = number of partitions
		self.max_exch_size = int(self.num_cells/6 - 1)

		self.node_to_net = {}
		for i, n in enumerate(self.nets):
			for nn in n:
				if nn in self.node_to_net.keys():
					self.node_to_net[nn].append(i)
				else:
					self.node_to_net[nn] = [i]

	# initializes population pool
	def init_population(self):
		self.pop = []
		for i in range(self.pop_count):
			new_sample = self.generate_random_sample()
			self.pop.append((new_sample, self.calc_cost(new_sample)))
		

	# runs the algorithm
	def run(self):
		start = time.time()
		self.init_population()

		self.it = 0
		self.swaps = 0
		self.paren_replace = 0

		prev_best = None
		results = []
		self.best_replace = 0
		print("[{:.2f}] Population Initialized".format(time.time() - start))

		while not self.stopping_cond():
			# select parents
			max_c = max([p[1] for p in self.pop])
			min_c = min([p[1] for p in self.pop])

			# apply fitness func and normalize to get probabilities for each 
			# gene
			fitness = [max_c - p[1] + (max_c - min_c)/3 for p in self.pop]
			fitness = [f/sum(fitness) for f in fitness]

			ps = np.random.choice(list(range(self.pop_count)), 
			                      2,       
			                      replace=False, p=fitness)

			# crossover: sample split locations and sort
			splits = sorted(
				np.random.choice(list(range(1, self.num_cells - 1)), 
				self.splits, 
				replace=False))

			offspring0 = ""
			offspring1 = ""

			# pad splits with bounds
			splits.insert(0, 0)
			splits.append(self.num_cells)

			# get pairs of (i, i+1) for bounds
			for i, v in enumerate(zip(splits[:-1], splits[1:])):
				parent = i%2
				# get segment of current split
				seg = self.pop[ps[parent]][0][v[0]:v[1]]
				offspring0 += seg 
				# flip segments of parent 1 for offspring 1
				offspring1 += seg if parent == 0 else self.complement(seg)

			# adjust/rebalance gene and apply local improvement
			offspring0 = self.local_improvement(self.adjust(offspring0))
			offspring1 = self.local_improvement(self.adjust(offspring1))
			o0_cost = self.calc_cost(offspring0)
			o1_cost = self.calc_cost(offspring1)
			ol = o0_cost if o0_cost <  o1_cost else o1_cost
			oh = o0_cost if o0_cost >= o1_cost else o1_cost

			ps = sorted([(x, self.pop[x][1]) for x in ps], key=lambda x: x[1])

			# pick worst genes from population for replacement candidates
			replace = sorted(enumerate(self.pop), 
			                 key=lambda x: x[1][1], 
							 reverse=True)[0:2]

			# case 1: both offsprings are better than both parents
			if ol < ps[0][1] and oh < ps[1][1]:
				# replace both parents
				self.pop[ps[0][0]] = (offspring0, o0_cost)
				self.pop[ps[1][0]] = (offspring1, o1_cost)
				self.paren_replace += 2
			
			# case 2: both offsprings are worse than both parents
				# replace worst genes in population
				self.pop[replace[0][0]] = (offspring0, o0_cost)
				self.pop[replace[1][0]] = (offspring1, o1_cost)

			# case 3: others
				# replace worse parent and worst gene in population
				self.pop[ps[1][0]] = (offspring0, o0_cost)
				self.pop[replace[0][0]] = (offspring1, o1_cost)
				self.paren_replace += 1
		
			new_best = min(self.pop, key=lambda x: x[1])
			results.append(new_best)
			
			# log best solution
			print("[{:.2f}] Best solution so far: {} (it={})".format(
				time.time() - start, 
				new_best, 
				self.it))
			if prev_best == None or prev_best > new_best[1]:
				prev_best = new_best[1]
				self.best_replace += 1
			self.it += 1

		print("[{:.2f}] Algorithm ended (Stopping condition: {})".format(
			time.time() - start, 
			"MAX_ITER REACHED" if self.it >= 3000 else 
			"BEST SOL CONVERAGED AFTER {} ITS".format(self.it)))
		print("[{:.2f}] Best solution: {}".format(
			time.time() - start, min(self.pop, key=lambda x: x[1])))
		
		self.run_time = time.time() - start
		
		if not results:
			results.append(min(self.pop, key=lambda x: x[1]))
		self.print_summary()
		return results

	def print_summary(self):
		print("Num Nodes: {}".format(self.num_cells))
		print("Avg Degree: {:.2f}".format(
			sum([len(v) for _, v in self.node_to_net.items()])/self.num_cells))
		print("Num Nets: {}".format(self.num_nets))
		print("Avg Net Size: {:.2f}".format(
			sum([len(x) for x in self.nets])/self.num_nets))
		print("Num Local Improvement Swaps: {} (Avg: {:.2f})".format(
			self.swaps, self.swaps/self.it if self.it else 0))
		print("Num Parent Replacements: {}".format(self.paren_replace))
		print("Num Best Replacements: {}".format(self.best_replace))
		print("Total Runtime: {:.2f} sec ({:.4f} sec/it)".format(
			self.run_time, self.run_time/self.it if self.it else 0))
	
	def stopping_cond(self):
		if self.it >= 3000:
			return True
		quality = sorted([x[1] for x in self.pop])
		return quality[0] == quality[int(len(quality)*0.8)]

	# given a potentially imbalanced gene, balance it
	def adjust(self, sol):
		sol = list(sol)

		count = self.count(sol)
		imbalance = count[0] - count[1]
		
		# if solution is already balanced, return
		if abs(imbalance) <= 1:
			return "".join(sol)

		# more L than R
		elif imbalance > 1:
			flipped = "L"
		else:
			flipped = "R"

		flips = abs(imbalance)//2
		# start from random location
		start = random.randint(0, len(sol) - 1)

		for i in range(len(sol)):
			# wrap around
			checked = (start + i)%len(sol)
			if sol[checked] == flipped:
				sol[checked] = self.complement(sol[checked])
				flips -= 1
				if flips == 0:
					return "".join(sol)
		assert(False)

	# apply Fiduccia-Mattheyes algorithm on solution for local improvement
	def local_improvement(self, sol):
		lgains = {}
		rgains = {}

		# calculate gains and split into left and right lists
		for i, _ in enumerate(sol):
			if sol[i] == "L":
				lgains[i] = self.gain(sol, i)
			else:
				rgains[i] = self.gain(sol, i)

		lswap = []
		rswap = []
		running_gain = []
		updated_sol = sol

		for i in range(self.max_exch_size):
			# get top 2 gains from each gain list
			ltop = sorted(lgains.items(), key=lambda x: x[1], reverse=True)[0:2]
			rtop = sorted(rgains.items(), key=lambda x: x[1], reverse=True)[0:2]

			best_joint = None
			best_pair  = None

			# try all possibilities
			for l in ltop:
				for r in rtop:
					# calc joint gain by summing up individual gains and 
					# subtracting gains that are erased
					joint_gain = (l[1]
					              + r[1]
								  - self.delta(updated_sol, l[0], r[0]))

					# verification: compare with brute-force solution
					# joint gain + current cost must be equal to cost with swap
					# assert(abs((self.cost_verif(updated_sol) - joint_gain)
					#             - self.cost_verif(
					# 				self.flip(updated_sol, 
					#                           [l[0], r[0]]))) < 1e-6)

					# keep track of best swap
					if best_joint == None or best_joint < joint_gain:
						best_joint = joint_gain
						best_pair = (l[0], r[0])
			
			# keep track of running gain after this iteration
			running_gain.append(best_joint 
			                    + (running_gain[-1] if running_gain else 0))

			# keep track of swapped pair and remove pair from gain list
			lswap.append(best_pair[0])
			rswap.append(best_pair[1])

			del lgains[best_pair[0]]
			del rgains[best_pair[1]]

			updated_sol = self.flip(updated_sol, best_pair)

			# update gains
			# if nets are treated as hyper-edges, the gain after the swap 
			# cannot be trivially determined, so just recalculate the gain of 
			# each neighboring node
			if self.hyper:
				neighbors = set()
				for p in best_pair:
					for n in self.node_to_net[p]:
						neighbors = neighbors.union(set(self.nets[n]))
				for n in neighbors:
					if n not in lswap and n not in rswap:
						gain_dict = lgains if updated_sol[n] == "L" else rgains
						gain_dict[n] = self.gain(updated_sol, n)
			# if nets are treated as cliques, the gain after the swap can be 
			# calculated by adjusting it based on the weight of the edge
			# the idea is the same as described in the paper but we iterate on 
			# the edges instead of the neighboring nodes (this also means the 
			# the delta function is not reused)
			else:
				for p in best_pair:
					for n in self.node_to_net[p]:
						for nn in self.nets[n]:
							if nn not in lswap and nn not in rswap:
								gain_dict = lgains if \
									updated_sol[nn] == "L" else rgains
								if updated_sol[nn] == updated_sol[p]:
									gain_dict[nn] -= 2/len(self.nets[n])
								else:
									gain_dict[nn] += 2/len(self.nets[n])
	
		# find best running gain
		max_gain = max(enumerate(running_gain), key=lambda x: x[1])

		# construct list of indices to flip
		flip_list = lswap[:max_gain[0]+1]
		flip_list.extend(rswap[:max_gain[0]+1])
		self.swaps += max_gain[0]

		return self.flip(sol, flip_list)

	# check if the given index is the last node of the given net to be on a 
	# separate partition (conversely, return if the given node is the only node 
	# out of the given net to be in a different partition)
	def check_last(self, sol, index, neighbors):
		return reduce(lambda x, y: x and y,
		              [x == index
					   or sol[x] != sol[index] for x in neighbors])

	# calculate the gain of each node, if nets are treated as hyper-edges
	def gain_hyper(self, sol, index):
		hyper_edges = self.node_to_net[index]
		gain = 0
		for e in hyper_edges:
			neighbors = self.nets[e]

			# gain is positive one if this is the last node to be moved to a 
			# separate partition for the given net
			gain += self.check_last(sol, index, neighbors)

			# gain is negative one if all nodes of a given net are already in 
			# the same partition
			gain -= reduce(lambda x, y: x and y, 
			              [sol[x] == sol[index] for x in neighbors])
		return gain

	# calculate the change in gain in the presence of a swap, if nets are  
	# treated as hyper-edges
	def delta_hyper(self, sol, a, b):
		intersect = list(set(self.node_to_net[a]) & set(self.node_to_net[b]))
		delta = 0

		# for each intersecting nets, check if a or b would have contributed to 
		# the gain. If it would, remove the gain since a direct swap of those 
		# would not lead to any gain (a becomes the last node in the other   
		# partition and vice-vera)
		for i in intersect:
			delta += self.check_last(sol, a, self.nets[i])
			delta += self.check_last(sol, b, self.nets[i])
		return delta 

	# calculate the gain of a given node, assuming the nets are treated as 
	# cliques
	def gain_clique(self, sol, index):
		nets = self.node_to_net[index]

		gain = 0
		for n in nets:
			neighbors = self.nets[n]
			w = 1/len(neighbors)
			for nn in neighbors:
				if nn != index:
					if sol[nn] == sol[index]:
						gain -= w
					else:
						gain += w
		return gain

	# calculate the change in gain in the presence of a swap, if the nets are 
	# treated as hyper-edges
	def delta_clique(self, sol, a, b):
		intersect = list(set(self.node_to_net[a]) & set(self.node_to_net[b]))

		# for each net that contains both of the swapped nodes, record the  
		# weights of the edge. This gain will be negated by the swap (2x, 1 per 
		# node)
		return 2*sum([1/len(self.nets[x]) for x in intersect])

	# calculate the cost of the given partition, assuming each net is a clique 
	# (via brute force, for verification)
	def calc_cost_clique(self, curr_sol):
		cost = 0
		for n in self.nets:
			w = 1/len(n)
			# treat each net as a clique, where there is a edge between each 
			# node within the net
			for p in itertools.combinations(n, 2):
				if curr_sol[p[0]] != curr_sol[p[1]]:
					cost += w
		return cost

	# calculate the cost of the given partition (via brute-force)
	def calc_cost(self, curr_sol):
		cost = 0
		for n in self.nets:
			side = None
			for nn in n:
				if nn < len(curr_sol):
					if side == None:
						side = curr_sol[nn]
					elif side != curr_sol[nn]:
						cost += 1
						break
		return cost

	# generate random, balanced partition
	def generate_random_sample(self):
		sol = ""
		l = 0
		r = 0
		balance = math.ceil(self.num_cells/2)

		for i in range(self.num_cells):
			if l < balance and r < balance:
				if random.uniform(0,1) > 0.5:
					sol += "R"
					r   += 1
				else:
					sol += "L"
					l   += 1
			elif l == balance:
				sol += "R"
				r   += 1
			elif r == balance:
				sol += "L"
				l   += 1
			else:
				assert(False)
		return sol

	# count the number of nodes in each partition for a given partition solution
	def count(self, sol):
		l = len(list(filter(lambda x: x == "L", sol)))
		return (l, len(sol) - l)
			
	# function flips the solution at the given indices
	def flip(self, sol, indices):
		sol = list(sol)

		for i in indices:
			sol[i] = self.complement(sol[i])

		return "".join(sol)

	# Function flips the given solution
	def complement(self, flipped_sol):
		flipped = ""

		for s in flipped_sol:
			flipped += "L" if s == "R" else "R"
		return flipped