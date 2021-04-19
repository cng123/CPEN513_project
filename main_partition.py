import random
import argparse
import math
from genetic_partition import GeneticPartition

def parse_benchmark(f):
	nets = [] 
	for i, l in enumerate(f.readlines()):
		if i == 0:
			sp = l.split(" ")

			num_cells = int(sp[0])

			if len(sp) > 2:
				num_rows  = int(sp[2])
				num_cols  = int(sp[3])
			else:
				num_rows = None
				num_cols = None
		else:
			net = []
			for i, ll in enumerate(l.strip().split(" ")):
				if i > 0:
					net.append(int(ll))
			if net:
				nets.append(net)
	return {"num_cells" : num_cells, "dim" : (num_rows, num_cols), "nets" : nets}

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--b", type=str)
	parser.add_argument("--net", type=str, choices=["hyper", "clique"], required=True)

	args = parser.parse_args()

	param = parse_benchmark(open(args.b))

	alg = GeneticPartition(param, 50, args.net)
	alg.run()
