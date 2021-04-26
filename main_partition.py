import random
import argparse
import math
import tkinter as tk
from genetic_partition import GeneticPartition
from GUI_partition import Renderer
from GUI_heatmap import HeatmapRenderer 
import matplotlib.pyplot as plt

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
	parser.add_argument(
		"--net", 
		type=str, 
		choices=["hyper", "clique"], 
		required=True)
	parser.add_argument(
		"--gui",
		action="store_true")
	parser.add_argument(
		"--no_local",
		action="store_true")

	args = parser.parse_args()

	param = parse_benchmark(open(args.b))

	alg = GeneticPartition(param, 50, args.net, args.no_local)
	sol = alg.run()

	if args.gui:
		top = tk.Tk()
		r = Renderer(top)
		r.pack()
		r.load(param["num_cells"], param["nets"], [x[0] for x in sol])
		top.mainloop()

		top = tk.Tk()
		h = HeatmapRenderer(top)
		h.load([x[1] for x in sol])
		h.pack()
		top.mainloop()

		fig, ax = plt.subplots(1,1)
		ax.set_title("Cost of Best Gene over Iterations")
		ax.set_xlabel("Iterations")
		ax.set_ylabel("Cost")

		if len(sol) > 1:
			ax.plot([x[0][1] for x in sol])
		# if no iterations are run, render scatter for single datapoint
		else:
			ax.scatter([0], [sol[0][0][1]])
		plt.show()
