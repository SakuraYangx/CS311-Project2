import numpy as np
import time
import sys
import queue
from random import *

start_time = time.time()
# filename = sys.argv[1]
# random_seed = sys.argv[5]
# f = open(filename, encoding='utf-8')
f = open('D:\python\CS311\Project\project2\CARP\CARP_samples\plaindata2.dat', encoding='utf-8')
lines = []
paraLines = []
dataLines = []
for line in f:
    s = line.strip().split('\n')
    lines.append(s)
f.close()

paraLines = lines[0:8]
dataLines = lines[9:-1]
for i in range(len(paraLines)):
    paraLines[i] = paraLines[i][0].split(' : ')

NAME = paraLines[0][1]
VERTICES = int(paraLines[1][1])
DEPOT = int(paraLines[2][1])
REQUIRED_EDGES = int(paraLines[3][1])
NON_REQUIRED_EDGES = int(paraLines[4][1])
VEHICLES = int(paraLines[5][1])
CAPACITY = int(paraLines[6][1])
TOTAL_COST_OF_REQUIRED_EDGES = int(paraLines[7][1])

edges = []
required_edges = []
costs = []
demands = []
for line in dataLines:
    tmp = line[0].split()
    edges.append((int(tmp[0]), int(tmp[1])))
    costs.append(int(tmp[2]))
    demands.append(int(tmp[3]))
    if int(tmp[3]) > 0:
        required_edges.append(((int(tmp[0]), int(tmp[1])), (int(tmp[2]), int(tmp[3]))))

cost_mat = np.zeros((VERTICES, VERTICES)) + sys.maxsize
demand_mat = np.zeros((VERTICES, VERTICES))
for i in range(len(edges)):
    edge = edges[i]
    v1 = edge[0]
    v2 = edge[1]
    cost = costs[i]
    demand = demands[i]
    cost_mat[v1 - 1][v2 - 1] = cost
    cost_mat[v2 - 1][v1 - 1] = cost
    demand_mat[v1 - 1][v2 - 1] = demand
    demand_mat[v2 - 1][v1 - 1] = demand


def dijkstra(start):
    distance_arr = np.zeros(VERTICES) + sys.maxsize
    prev = np.zeros(VERTICES, dtype=int) - 1
    flag_arr = np.zeros(VERTICES)
    distance_arr[start] = 0
    q = queue.PriorityQueue()
    q.put((0, start))
    while not q.empty():
        tmp_min = q.get()
        if flag_arr[tmp_min[1]] == 1:
            continue
        flag_arr[tmp_min[1]] = 1
        adjacent_nodes = np.argwhere(cost_mat[tmp_min[1]] < sys.maxsize)
        for i in adjacent_nodes:
            index = i[0]
            if prev[index] != -1:
                if distance_arr[index] > cost_mat[tmp_min[1]][index] + distance_arr[tmp_min[1]]:
                    distance_arr[index] = cost_mat[tmp_min[1]][index] + distance_arr[tmp_min[1]]
                    prev[index] = tmp_min[1]
                    q.put((distance_arr[index], index))
            elif prev[index] == -1 and index != start:
                prev[index] = tmp_min[1]
                distance_arr[index] = cost_mat[tmp_min[1]][index] + distance_arr[prev[index]]
                q.put((distance_arr[index], index))
    return distance_arr, prev


dist_mat = []
sorted_dists = []
for i in range(VERTICES):
    dist, _ = dijkstra(i)
    dist_mat.append(dist)
    ids = range(VERTICES)
    d = dict(zip(ids, dist))
    sorted_dist = sorted(d.items(), key=lambda d: d[1])
    sorted_dists.append(sorted_dist)


def path_scanning():
    paths = []
    path_costs = []
    required_edges_copy = required_edges.copy()
    total_demand = np.sum(demands)
    while total_demand > 0:
        cur_end = DEPOT - 1
        load = 0
        cur_path_cost = 0
        path = []
        while load < CAPACITY:
            remain_load = CAPACITY - load
            avail_edges = [edge for edge in required_edges_copy if edge[1][1] <= remain_load]
            if len(avail_edges) == 0:
                break
            min_dis = sys.maxsize
            pick_edge = None
            for tmp_edge in avail_edges:
                if min_dis > min(dist_mat[cur_end][tmp_edge[0][0] - 1], dist_mat[cur_end][tmp_edge[0][1] - 1]):
                    min_dis = min(dist_mat[cur_end][tmp_edge[0][0] - 1], dist_mat[cur_end][tmp_edge[0][1] - 1])
                    pick_edge = tmp_edge
            required_edges_copy.remove(pick_edge)
            if dist_mat[cur_end][pick_edge[0][0] - 1] < dist_mat[cur_end][pick_edge[0][1] - 1]:
                path.append((pick_edge[0][0], pick_edge[0][1]))
                load += pick_edge[1][1]
                cur_path_cost += dist_mat[cur_end][pick_edge[0][0] - 1]
                cur_path_cost += pick_edge[1][0]
                total_demand -= pick_edge[1][1]
                cur_end = pick_edge[0][1] - 1
            else:
                path.append((pick_edge[0][1], pick_edge[0][0]))
                load += pick_edge[1][1]
                cur_path_cost += dist_mat[cur_end][pick_edge[0][1] - 1]
                cur_path_cost += pick_edge[1][0]
                total_demand -= pick_edge[1][1]
                cur_end = pick_edge[0][0] - 1
        paths.append(path)
        cur_path_cost += dist_mat[cur_end][DEPOT - 1]
        path_costs.append(cur_path_cost)

    return paths, path_costs


def output(paths, total_cost):
    s = 's '
    for path in paths:
        s += '0,'
        for e in path:
            s += e.__str__() + ','
        s += '0,'

    s = s.replace(', ', ',')
    s = s[:-1]
    print(s)
    q = 'q ' + str(int(total_cost))
    print(q)


paths, path_costs = path_scanning()
total_cost = np.sum(path_costs)
output(paths, total_cost)
run_time = (time.time() - start_time)
