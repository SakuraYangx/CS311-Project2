import numpy as np
import time
import sys
import queue
import random
from threading import Thread

start_time = time.time()
filename = sys.argv[1]
TERMINATION_TIME = float(sys.argv[3])
TERMINATION_TIME -= 5
RANDOM_SEED = float(sys.argv[5])
random.seed(RANDOM_SEED)
f = open(filename, encoding='utf-8')
# TERMINATION_TIME = 30
# TERMINATION_TIME -= 5
# f = open('D:\python\CS311\Project\project2\CARP\CARP_samples\egl-s1-A.dat', encoding='utf-8')
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


class MyThread(Thread):
    def __init__(self, func, args):
        '''
        :param func: 可调用的对象
        :param args: 可调用对象的参数
        '''
        Thread.__init__(self)
        self.func = func
        self.args = args
        self.result = None

    def run(self):
        self.result = self.func(*self.args)

    def getResult(self):
        return self.result


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


def generatic(total_time):
    def path_scanning(avail_time, ubtrial):
        """

        :param avail_time: time used to general population
        :return: route_set, cost_set
        """
        time1 = time.time()
        time2 = time.time()
        route_set = []  # population
        trial_time = 0
        while avail_time - (time2 - time1) > 0 and trial_time < ubtrial:
            # while trial_time < ubtrial:
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
            if paths not in route_set:
                route_set.append([paths, np.sum(path_costs)])
                # trial_time += 1 #delete!!!!!!!
            else:
                trial_time += 1
            time2 = time.time()
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
            total += 1 / population[i][1]
        probability = [1 / (population[j][1] * total) for j in range(len(population))]
        p = np.array(probability)
        index_list = list(range(len(population)))
        for i in range(r):
            index = np.random.choice(index_list, p=p.ravel())
            parent.append(population[index])
        return parent

    def crossover(paths, ubtrial):
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
                    continue
            else:
                continue
        return 0

    def mutation(paths, Pf=0.1, Psi=0.2):
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
                    return 0
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

    begin_time = time.time()
    route_set = path_scanning(total_time / 5, 50)
    population = route_set
    size = len(population)
    while time.time() - begin_time < total_time and size != 1:
        # while size != 1:
        newGen = select(int(size / 2), population)
        for solution in newGen:
            solution[1] += crossover(solution[0], 20)
            solution[0] = list(filter(None, solution[0]))
            solution[1] += mutation(solution[0])
            solution[0] = list(filter(None, solution[0]))
        population = newGen
        size = len(population)
    return population


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


def main():
    # multi_thread
    t1 = MyThread(generatic, (TERMINATION_TIME,))
    t2 = MyThread(generatic, (TERMINATION_TIME,))
    t3 = MyThread(generatic, (TERMINATION_TIME,))
    t4 = MyThread(generatic, (TERMINATION_TIME,))
    t5 = MyThread(generatic, (TERMINATION_TIME,))
    t6 = MyThread(generatic, (TERMINATION_TIME,))
    t7 = MyThread(generatic, (TERMINATION_TIME,))
    t8 = MyThread(generatic, (TERMINATION_TIME,))
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()
    t6.start()
    t7.start()
    t8.start()
    t1.join()
    t2.join()
    t3.join()
    t4.join()
    t5.join()
    t6.join()
    t7.join()
    t8.join()
    t1_result = calTotalCost(t1.getResult()[0][0])
    t2_result = calTotalCost(t2.getResult()[0][0])
    t3_result = calTotalCost(t3.getResult()[0][0])
    t4_result = calTotalCost(t4.getResult()[0][0])
    t5_result = calTotalCost(t5.getResult()[0][0])
    t6_result = calTotalCost(t6.getResult()[0][0])
    t7_result = calTotalCost(t7.getResult()[0][0])
    t8_result = calTotalCost(t8.getResult()[0][0])
    min_result = t1_result
    cor_solution = t1.getResult()[0][0]
    if t2_result < min_result:
        min_result = t2_result
        cor_solution = t2.getResult()[0][0]
    if t3_result < min_result:
        min_result = t3_result
        cor_solution = t3.getResult()[0][0]
    if t4_result < min_result:
        min_result = t4_result
        cor_solution = t4.getResult()[0][0]
    if t5_result < min_result:
        min_result = t5_result
        cor_solution = t5.getResult()[0][0]
    if t6_result < min_result:
        min_result = t6_result
        cor_solution = t6.getResult()[0][0]
    if t7_result < min_result:
        min_result = t7_result
        cor_solution = t7.getResult()[0][0]
    if t8_result < min_result:
        min_result = t8_result
        cor_solution = t8.getResult()[0][0]
    output(cor_solution, min_result)
    # print(t1.getResult()[0][1])
    # print(t2.getResult()[0][1])
    # print(t3.getResult()[0][1])
    # print(t4.getResult()[0][1])
    # print(t5.getResult()[0][1])
    # print(t6.getResult()[0][1])
    # print(t7.getResult()[0][1])
    # print(t8.getResult()[0][1])


# print(generatic(TERMINATION_TIME))


if __name__ == "__main__":
    main()
