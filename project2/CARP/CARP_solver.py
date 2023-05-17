import numpy as np
import time
import sys
import queue
import random
import multiprocessing

start_time = time.time()
# filename = sys.argv[1]
# TERMINATION_TIME = float(sys.argv[3])
# TERMINATION_TIME -= 5
# RANDOM_SEED = float(sys.argv[5])
# random.seed(RANDOM_SEED)
# f = open(filename, encoding='utf-8')
TERMINATION_TIME = 15
TERMINATION_TIME -= 5
RANDOM_SEED = 1512
f = open('D:\python\CS311\Project\project2\CARP\CARP_samples\\val7A.dat', encoding='utf-8')
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


# crossover_Pro = 0.4 #crossover
# flip_pro = 0.3 #flip
# sig_in_pro = 0.4 #single-insert
select_scale = 10 #run-select
run_time = 0.95 #run-time
ubtrial = 10 #path_scanning
scanning_time = 10 #generation population


class MyProcess(multiprocessing.Process):
    def __init__(self, usedTime, pi, q, crossover_Pro, flip_pro, sig_in_pro):
        super(MyProcess, self).__init__()
        # self.population = population
        self.usedTime = usedTime
        self.pi = pi + 1
        self.q = q
        self.crossover_Pro = crossover_Pro
        self.flip_pro = flip_pro
        self.sig_in_pro = sig_in_pro

    def run(self):
        time_begin_run = time.time()
        avail_time = (TERMINATION_TIME - self.usedTime)*run_time
        population = path_scanning(avail_time/5, 20)
        random.seed(RANDOM_SEED + self.pi)
        best_Solution = None
        local_min_cost = sys.maxsize
        mutation_gen = 0
        size = len(population)
        while time.time() - time_begin_run < avail_time*4/5:
            newGen = select(int(size / select_scale), population)
            mutation_gen += 1
            Pbf = 1/mutation_gen
            for solution in newGen:
                solution[1] += crossover(solution[0], 30-3*self.pi, Pbf, self.crossover_Pro**(0.99*mutation_gen))
                solution[0] = list(filter(None, solution[0]))
                solution[1] += mutation(solution[0], Pbf, self.flip_pro**(0.99*mutation_gen), self.sig_in_pro**(0.99*mutation_gen))
                solution[0] = list(filter(None, solution[0]))
                if solution[1] < local_min_cost:
                    local_min_cost = solution[1]
                    best_Solution = solution
        self.q.put(best_Solution)


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
for i in range(VERTICES):
    dist, _ = dijkstra(i)
    dist_mat.append(dist)


def calPathLoad(path):
    """

    :param path:
    :return:
    """
    load = 0
    for task in path:
        load += demand_mat[task[0] - 1][task[1] - 1]
    return load


def calEachCost(path):
    cost_sum = 0
    pos = DEPOT
    for e in path:
        cost_sum += dist_mat[pos - 1][e[0] - 1]
        cost_sum += cost_mat[e[0] - 1, e[1] - 1]
        pos = e[1]
    cost_sum += dist_mat[pos - 1][DEPOT - 1]
    return cost_sum


def calTotalCost(paths):
    cost_sum = 0
    for path in paths:
        cost_sum += calEachCost(path)
    return cost_sum


def path_scanning(avail_time, ubtrial):
    """

    :param avail_time: time used to general population
    :return: route_set, cost_set
    """
    time_begin_search = time.time()
    route_set = []  # population
    trial_times = 0
    while avail_time - (time.time() - time_begin_search) > 0 and trial_times < ubtrial:
        paths = []  # individual
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
                    elif min_dis == min(dist_mat[cur_end][tmp_edge[0][0] - 1],
                                        dist_mat[cur_end][tmp_edge[0][1] - 1]):
                        if bool(random.getrandbits(1)):
                            continue
                        else:
                            min_dis = min(dist_mat[cur_end][tmp_edge[0][0] - 1],
                                          dist_mat[cur_end][tmp_edge[0][1] - 1])
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
        if [paths, np.sum(path_costs)] in route_set:
            trial_times += 1
        else:
            route_set.append([paths, np.sum(path_costs)])
    return route_set


def select(r, population):
    """

    :param r: number of parents
    :param population: original sample_set
    :param cost_set:
    :return: parent_set
    """
    parent = []
    total = 0
    for i in range(len(population)):
        total += 1 / (population[i][1]**(1/3))
    probability = [1 / ((population[j][1]**(1/3)) * total) for j in range(len(population))]
    p = np.array(probability)
    index_list = list(range(len(population)))
    for i in range(r):
        index = np.random.choice(index_list, p=p.ravel())
        parent.append(population[index])
    return parent


def crossover(paths, ubtrial, Pbf, P):
    Rdc = random.random()
    if Rdc < P:
        if len(paths) - 2 <= 0:
            return 0
        trial_time = 0
        while trial_time <= ubtrial:
            trial_time += 1
            path1 = random.randint(0, len(paths) - 1)
            path2 = random.randint(0, len(paths) - 1)
            if paths[path1] == paths[path2]:
                continue
            x_split_pos = random.randint(0, len(paths[path1]) - 1)
            y_split_pos = random.randint(0, len(paths[path2]) - 1)
            new_path_1 = paths[path1][0:x_split_pos] + paths[path2][y_split_pos:]
            new_path_2 = paths[path2][0:y_split_pos] + paths[path1][x_split_pos:]
            if calPathLoad(new_path_1) <= CAPACITY and calPathLoad(new_path_2) <= CAPACITY:
                if x_split_pos > 0 and y_split_pos > 0:
                    cost_change = dist_mat[paths[path1][x_split_pos - 1][1] - 1][paths[path2][y_split_pos][0] - 1] + \
                                dist_mat[paths[path2][y_split_pos - 1][1] - 1][paths[path1][x_split_pos][0] - 1] - \
                                dist_mat[paths[path1][x_split_pos - 1][1] - 1][paths[path1][x_split_pos][0] - 1] - \
                                dist_mat[paths[path2][y_split_pos - 1][1] - 1][paths[path2][y_split_pos][0] - 1]
                elif x_split_pos == 0 and y_split_pos > 0:
                    cost_change = dist_mat[DEPOT - 1][paths[path2][y_split_pos][0] - 1] + \
                                dist_mat[paths[path2][y_split_pos - 1][1] - 1][paths[path1][x_split_pos][0] - 1] - \
                                dist_mat[DEPOT - 1][paths[path1][x_split_pos][0] - 1] - \
                                dist_mat[paths[path2][y_split_pos - 1][1] - 1][paths[path2][y_split_pos][0] - 1]
                elif x_split_pos > 0 and y_split_pos == 0:
                    cost_change = dist_mat[paths[path1][x_split_pos - 1][1] - 1][paths[path2][y_split_pos][0] - 1] + \
                                dist_mat[DEPOT - 1][paths[path1][x_split_pos][0] - 1] - \
                                dist_mat[paths[path1][x_split_pos - 1][1] - 1][paths[path1][x_split_pos][0] - 1] - \
                                dist_mat[DEPOT - 1][paths[path2][y_split_pos][0] - 1]
                else:
                    cost_change = 0
                if cost_change < 0:
                    temp = paths[path1][x_split_pos:]
                    paths[path1] = paths[path1][0:x_split_pos] + paths[path2][y_split_pos:]
                    paths[path2] = paths[path2][0:y_split_pos] + temp
                    return cost_change
                else:
                    r = random.random()
                    if r < Pbf:
                        temp = paths[path1][x_split_pos:]
                        paths[path1] = paths[path1][0:x_split_pos] + paths[path2][y_split_pos:]
                        paths[path2] = paths[path2][0:y_split_pos] + temp
                        return cost_change
                    else:
                        continue
            else:
                continue
    return 0


def mutation(paths, Pbf, Pf, Psi):
    """

    :param paths:
    :param Pf: probability to filp
    :param Psi: probability to single insertion
    :return:
    """

    def flip(paths):
        """

        :param paths:
        :return:
        """
        path = random.randint(0, len(paths) - 1)
        if len(paths[path]) == 1:
            return 0
        else:
            task = random.randint(0, len(paths[path]) - 1)
            if task == 0:
                cost_change = dist_mat[DEPOT - 1][paths[path][task][1] - 1] + \
                              dist_mat[paths[path][task][0] - 1][paths[path][task + 1][0] - 1] - \
                              dist_mat[DEPOT - 1][paths[path][task][0] - 1] - \
                              dist_mat[paths[path][task][1] - 1][paths[path][task + 1][0] - 1]
            elif task == len(paths[path]) - 1:
                cost_change = dist_mat[paths[path][task - 1][1] - 1][paths[path][task][1] - 1] + \
                              dist_mat[paths[path][task][0] - 1][DEPOT - 1] - \
                              dist_mat[paths[path][task - 1][1] - 1][paths[path][task][0] - 1] - \
                              dist_mat[paths[path][task][1] - 1][DEPOT - 1]
            else:
                cost_change = dist_mat[paths[path][task - 1][1] - 1][paths[path][task][1] - 1] + \
                              dist_mat[paths[path][task][0] - 1][paths[path][task + 1][0] - 1] - \
                              dist_mat[paths[path][task - 1][1] - 1][paths[path][task][0] - 1] - \
                              dist_mat[paths[path][task][1] - 1][paths[path][task + 1][0] - 1]
            if cost_change > 0:
                r = random.random()
                if r < Pbf:
                    paths[path][task] = tuple(reversed(paths[path][task]))
                    return cost_change
                else:
                    return 0
            else:
                paths[path][task] = tuple(reversed(paths[path][task]))
                return cost_change

    def single_insertion(paths):
        """

        :param paths:
        :return:
        """
        path1 = random.randint(0, len(paths) - 1)
        path2 = random.randint(0, len(paths) - 1)
        if path1 == path2:
            if len(paths[path1]) == 1:
                return 0
            else:
                task1 = random.randint(0, len(paths[path1]) - 1)
                task2 = random.randint(0, len(paths[path1]) - 1)
                if task1 == task2:
                    return 0
                else:
                    task_f = min(task1, task2)
                    task_l = max(task1, task2)
                    if task_f != 0 and task_l != len(paths[path1]) - 1:
                        cost_change = dist_mat[paths[path1][task_f - 1][1] - 1][paths[path1][task_l][0] - 1] + \
                                      dist_mat[paths[path1][task_f][1] - 1][paths[path1][task_l + 1][0] - 1] + \
                                      dist_mat[paths[path1][task_f][0] - 1][paths[path1][task_l][1] - 1] - \
                                      dist_mat[paths[path1][task_f - 1][1] - 1][paths[path1][task_f][0] - 1] - \
                                      dist_mat[paths[path1][task_l][1] - 1][paths[path1][task_l + 1][0] - 1] - \
                                      dist_mat[paths[path1][task_f][1] - 1][paths[path1][task_l][0] - 1]
                    elif task_f == 0 and task_l != len(paths[path1]) - 1:
                        cost_change = dist_mat[DEPOT - 1][paths[path1][task_l][0] - 1] + \
                                      dist_mat[paths[path1][task_f][1] - 1][paths[path1][task_l + 1][0] - 1] + \
                                      dist_mat[paths[path1][task_f][0] - 1][paths[path1][task_l][1] - 1] - \
                                      dist_mat[DEPOT - 1][paths[path1][task_f][0] - 1] - \
                                      dist_mat[paths[path1][task_l][1] - 1][paths[path1][task_l + 1][0] - 1] - \
                                      dist_mat[paths[path1][task_f][1] - 1][paths[path1][task_l][0] - 1]
                    elif task_f != 0 and task_l == len(paths[path1]) - 1:
                        cost_change = dist_mat[paths[path1][task_f - 1][1] - 1][paths[path1][task_l][0] - 1] + \
                                      dist_mat[paths[path1][task_f][1] - 1][DEPOT - 1] + \
                                      dist_mat[paths[path1][task_f][0] - 1][paths[path1][task_l][1] - 1] - \
                                      dist_mat[paths[path1][task_f - 1][1] - 1][paths[path1][task_f][0] - 1] - \
                                      dist_mat[paths[path1][task_l][1] - 1][DEPOT - 1] - \
                                      dist_mat[paths[path1][task_f][1] - 1][paths[path1][task_l][0] - 1]
                    else:
                        cost_change = dist_mat[DEPOT - 1][paths[path1][task_l][0] - 1] + \
                                      dist_mat[paths[path1][task_f][1] - 1][DEPOT - 1] + \
                                      dist_mat[paths[path1][task_f][0] - 1][paths[path1][task_l][1] - 1] - \
                                      dist_mat[DEPOT - 1][paths[path1][task_f][0] - 1] - \
                                      dist_mat[paths[path1][task_l][1] - 1][DEPOT - 1] - \
                                      dist_mat[paths[path1][task_f][1] - 1][paths[path1][task_l][0] - 1]
                    if cost_change < 0:
                        tmp = paths[path1][task_f]
                        paths[path1][task_f] = paths[path1][task_l]
                        paths[path1][task_l] = tmp
                        return cost_change
                    else:
                        r = random.random()
                        if r < Pbf:
                            tmp = paths[path1][task_f]
                            paths[path1][task_f] = paths[path1][task_l]
                            paths[path1][task_l] = tmp
                            return cost_change
                        else:
                            return 0
        else:
            if len(paths[path1]) == 1 and len(paths[path2]) == 1:
                return 0
            path_p = path1 if max(len(paths[path1]), len(paths[path2])) == len(paths[path1]) else path2
            path_n = path2 if max(len(paths[path1]), len(paths[path2])) == len(paths[path1]) else path1
            task1 = random.randint(0, len(paths[path_p]) - 1)
            task2 = random.randint(0, len(paths[path_n]) - 1)
            new_path_1 = paths[path_p][:task1] + paths[path_p][task1 + 1:]
            new_path_2 = paths[path_n][:task2] + [paths[path_p][task1]] + paths[path_n][task2:]
            if calPathLoad(new_path_1) > CAPACITY or calPathLoad(new_path_2) > CAPACITY:
                return 0
            if task1 != 0 and task2 != 0 and task1 != len(paths[path_p]) - 1:
                cost_change = dist_mat[paths[path_n][task2 - 1][1] - 1][paths[path_p][task1][0] - 1] + \
                              dist_mat[paths[path_n][task2][0] - 1][paths[path_p][task1][1] - 1] + \
                              dist_mat[paths[path_p][task1 - 1][1] - 1][paths[path_p][task1 + 1][0] - 1] - \
                              dist_mat[paths[path_p][task1 - 1][1] - 1][paths[path_p][task1][0] - 1] - \
                              dist_mat[paths[path_p][task1 + 1][0] - 1][paths[path_p][task1][1] - 1] - \
                              dist_mat[paths[path_n][task2 - 1][1] - 1][paths[path_n][task2][0] - 1]
            elif task1 == 0 and task2 != 0:
                cost_change = dist_mat[paths[path_n][task2 - 1][1] - 1][paths[path_p][task1][0] - 1] + \
                              dist_mat[paths[path_n][task2][0] - 1][paths[path_p][task1][1] - 1] + \
                              dist_mat[DEPOT - 1][paths[path_p][task1 + 1][0] - 1] - \
                              dist_mat[DEPOT - 1][paths[path_p][task1][0] - 1] - \
                              dist_mat[paths[path_p][task1 + 1][0] - 1][paths[path_p][task1][1] - 1] - \
                              dist_mat[paths[path_n][task2 - 1][1] - 1][paths[path_n][task2][0] - 1]
            elif task1 == len(paths[path_p]) - 1 and task2 != 0:
                cost_change = dist_mat[paths[path_n][task2 - 1][1] - 1][paths[path_p][task1][0] - 1] + \
                              dist_mat[paths[path_n][task2][0] - 1][paths[path_p][task1][1] - 1] + \
                              dist_mat[paths[path_p][task1 - 1][1] - 1][DEPOT - 1] - \
                              dist_mat[paths[path_p][task1 - 1][1] - 1][paths[path_p][task1][0] - 1] - \
                              dist_mat[DEPOT - 1][paths[path_p][task1][1] - 1] - \
                              dist_mat[paths[path_n][task2 - 1][1] - 1][paths[path_n][task2][0] - 1]
            elif task1 != 0 and task2 == 0 and task1 != len(paths[path_p]) - 1:
                cost_change = dist_mat[DEPOT - 1][paths[path_p][task1][0] - 1] + \
                              dist_mat[paths[path_n][task2][0] - 1][paths[path_p][task1][1] - 1] + \
                              dist_mat[paths[path_p][task1 - 1][1] - 1][paths[path_p][task1 + 1][0] - 1] - \
                              dist_mat[paths[path_p][task1 - 1][1] - 1][paths[path_p][task1][0] - 1] - \
                              dist_mat[paths[path_p][task1 + 1][0] - 1][paths[path_p][task1][1] - 1] - \
                              dist_mat[DEPOT - 1][paths[path_n][task2][0] - 1]
            elif task1 == 0 and task2 == 0:
                cost_change = dist_mat[DEPOT - 1][paths[path_p][task1][0] - 1] + \
                              dist_mat[paths[path_n][task2][0] - 1][paths[path_p][task1][1] - 1] + \
                              dist_mat[DEPOT - 1][paths[path_p][task1 + 1][0] - 1] - \
                              dist_mat[DEPOT - 1][paths[path_p][task1][0] - 1] - \
                              dist_mat[paths[path_p][task1 + 1][0] - 1][paths[path_p][task1][1] - 1] - \
                              dist_mat[DEPOT - 1][paths[path_n][task2][0] - 1]
            else:
                cost_change = dist_mat[DEPOT - 1][paths[path_p][task1][0] - 1] + \
                              dist_mat[paths[path_n][task2][0] - 1][paths[path_p][task1][1] - 1] + \
                              dist_mat[paths[path_p][task1 - 1][1] - 1][DEPOT - 1] - \
                              dist_mat[paths[path_p][task1 - 1][1] - 1][paths[path_p][task1][0] - 1] - \
                              dist_mat[DEPOT - 1][paths[path_p][task1][1] - 1] - \
                              dist_mat[DEPOT - 1][paths[path_n][task2][0] - 1]
            if cost_change > 0:
                r = random.random()
                if r < Pbf:
                    paths[path_p] = new_path_1
                    paths[path_n] = new_path_2
                    return cost_change
                else:
                    return 0
            else:
                paths[path_p] = new_path_1
                paths[path_n] = new_path_2
                return cost_change

    r = random.random()
    if r < Pf:
        cost_change = flip(paths)
    elif Pf <= r < Psi:
        cost_change = single_insertion(paths)
    else:
        cost_change = 0
    return cost_change


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


if __name__ == '__main__':
    P = 8
    q = multiprocessing.Manager().Queue()
    jobs = []
    for proc in range(1):
        if proc == P-1:
            sub_Process = MyProcess(time.time()-start_time, proc, q, 0.9, 0.4, 0.7)
        else:
            sub_Process = MyProcess(time.time()-start_time, proc, q, 0.4, 0.3, 0.4)
        sub_Process.start()
        jobs.append(sub_Process)
    for p in jobs:
        p.join()
    results = [q.get() for j in jobs]
    best_Solution = None
    min_cost = sys.maxsize
    # for result in results:
    #     print(calTotalCost(result[0]))
    for result in results:
        if calTotalCost(result[0]) < min_cost:
            min_cost = calTotalCost(result[0])
            best_Solution = result[0]
    output(best_Solution, min_cost)


