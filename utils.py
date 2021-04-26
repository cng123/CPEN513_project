import random
from math import floor
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import colors
from tabulate import tabulate

def preprocess(file):
    #read file as lines
    f = open(file, "r").readlines()

    curr_line = f[0].split()
    ny, nx = int(curr_line[2]), int(curr_line[3])
    cell_num = int(curr_line[0])
    net_num = int(curr_line[1])
    nets = []

    #save nets info
    for i in range(1, 1+net_num):
        curr_net = f[i].split()
        nets.append(set([int(i) for i in curr_net][1:]))

    return ny, nx, cell_num, nets

def print_chromosome(chromosome):
    cell_num = []
    x = []
    y = []
    for gene in chromosome:
        if gene:
            cell_num.append(gene.cell_num)
            x.append(gene.x)
            y.append(gene.y)
        else:
            cell_num.append(-1)
            x.append(-1)
            y.append(-1)
    print(tabulate([x, y], headers=cell_num))
    print("")

def plot(avg_cost, min_cost, max_cost, diff):
    plt.title('cost over generations')
    plt.plot(avg_cost)

    x = np.arange(len(avg_cost))
    y_bot = min_cost
    y_dif = diff
    plt.bar(x, y_dif, bottom=y_bot)

    plt.show(block=False)
    plt.pause(1)
    plt.close()