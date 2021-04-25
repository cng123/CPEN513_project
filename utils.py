import random
from math import floor
import numpy as np
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

def initialize(ny, nx, cell_num, nets):
    cells_pos = {}
    nets_info = []
    cells_info = [[] for i in range(cell_num)]
    grid = np.full((ny,nx), -1)
    cost = 0

    # randomly generate initial positions
    random.seed(10)
    pos = random.sample(range(ny*nx), cell_num)

    # decode pos into x, y coordinates, and update on grid
    for i,n in enumerate(pos):
        cells_pos[i] = [floor(n/nx), n%nx]
        grid[floor(n/nx)][n%nx] = i

    # create net objects for each net, and save initial information
    for i,net in enumerate(nets):
        pos = cells_pos[net[0]]
        net_obj = Net(pos, net[0], net)
        cells_info[net[0]].append(i)
        for c in net[1:]:
            # add all cells of net to net object
            net_obj.add_cell(cells_pos[c], c)
            # save the nets each cell is in
            cells_info[c].append(i)
        # calculate initial cost
        cost += net_obj.update_cost()
        nets_info.append(net_obj)

    return cells_pos, nets_info, cells_info, grid, cost

def print_chromosome(chromosome):
    cell_num = []
    x = []
    y = []
    for gene in chromosome:
        cell_num.append(gene.cell_num)
        x.append(gene.x)
        y.append(gene.y)
    print(tabulate([x, y], headers=cell_num))
    print("")