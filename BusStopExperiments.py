import itertools
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import random
import time
import json
import math
import matplotlib.pyplot as plt
import seaborn as sns




def closest_index(stops, x):
    ret = []
    i = 0
    while i < len(stops) and stops[i] < x:
        i += 1
    ret = -1
    dist_ret = 1000
    if i > 0:
        ret = i-1
        dist_ret = abs(stops[i-1]- x)
    if i < len(stops):
        if stops[i] == x:
            return i
        if dist_ret > abs(stops[i]- x):
            ret = i
    return ret

def possible_stops(list1, x):
    ret = []
    i = 0
    while i < len(list1) and list1[i] < x:
        i += 1

    if i > 0:
        ret.append(list1[i-1])
    if i < len(list1):
        if list1[i] == x:
            return [x]
        ret.append(list1[i])
    return ret


def get_cost(agent, sol, alpha):
    possible_s = possible_stops(sol, agent[0])
    possible_t = possible_stops(sol, agent[1])

    cost = agent[1] - agent[0]
    for s in possible_s:
        for t in possible_t:
            cost = min(cost, abs(s-agent[0]) + abs(t-agent[1]) + alpha*abs(s-t))
    return cost

def generate_improving_pairs(agents, stops, sol, alpha):
    n = len(agents)
    m = len(stops)
    k = len(sol)
    improving_pairs = []
    for i in range(n):
        improving_pairs.append([])
    # print(improving_pairs)
    costs = [get_cost(a, sol, alpha) for a in agents]
    # print(costs)
    for s, t in itertools.product(range(m), range(m)):
        if s < t:
            # print(stops[s], stops[t])
            temp_costs = [get_cost(a, [stops[s], stops[t]], alpha) for a in agents]
            for i in range(n):
                if temp_costs[i] < costs[i]:
                    improving_pairs[i].append((s, t))
                    # print(i, stops[s], stops[t], improving_pairs)

    return improving_pairs


def check_JR(improving_pairs, l):
    freq = {}
    for a in improving_pairs:
        for p in a:
            if p in freq:
                freq[p] += 1
            else:
                freq[p] = 1
    for p in freq:
        if freq[p] >= l:
            return False
    return True


def check_in_the_core(agents, stops, sol, alpha):
    n = len(agents)
    m = len(stops)
    k = len(sol)
    improving_pairs = generate_improving_pairs(agents, stops, sol, alpha)

    model = gp.Model()
    model.params.LogToConsole = 0
    x = model.addVars(n, vtype=GRB.BINARY, name = "x")
    y = model.addVars(m, vtype=GRB.BINARY, name = "y")
    z = model.addVars(m* m, vtype=GRB.BINARY, name = "z")

    model.addConstrs(z[m*i+j] <= y[i] for i in range(m) for j in range(m))
    model.addConstrs(z[m*i+j] <= y[j] for i in range(m) for j in range(m))
    model.addConstrs(x[i] <= gp.quicksum(z[m*t[0]+t[1]] for t in improving_pairs[i]) for i in range(n))

    model.addConstr(gp.quicksum(x[i] for i in range(n)) >= gp.quicksum(y[j] for j in range(m))*n/k)
    # model.addConstr(gp.quicksum(x[i] for i in range(n)) >= 1)

    model.setObjective(gp.quicksum(x[i] for i in range(n)), GRB.MAXIMIZE)
    model.optimize()

    if model.ObjVal == 0:
        return True, True
    else:
        if check_JR(improving_pairs, math.ceil(2*n/k)) == False:
            return False, False
        else:
            return False, True


def generate_random_instance(n, m, max_stop = 100):
    # stops: m random stops uniformly from [1 .. max_stop]
    # agents: n random pair of points from [1 .. max_stop]
    #         for each agent I select his left and right terminals uniformly at random from [stops]

    stops = sorted(random.sample(range(1, max_stop+1), m))
    agents = []

    for i in range(n):
        agents.append(sorted(random.sample(stops, 2)))

    return agents, stops


def benchmark(n, stops, k):
    m = len(stops)

    solution = [] # list of lenght k
    max_terminal = max(stops)
    temp = [0]*(max_terminal+1)
    y = 0
    remaining_agents = 2*n
    while remaining_agents > 0 and y < m:
        if y == 0:
            temp[stops[0]] += min(remaining_agents, math.floor(n/max_terminal*stops[0]))
            remaining_agents -= min(remaining_agents, math.floor(n/max_terminal*stops[0]))
        else:
            temp[stops[y]] += min(remaining_agents, math.floor(n/max_terminal*(stops[y]-stops[y-1])/2))
            remaining_agents -= min(remaining_agents, math.floor(n/max_terminal*(stops[y]-stops[y-1])/2))
        if y == m-1:
            temp[stops[y]] += min(remaining_agents, math.floor(n/max_terminal*(100-stops[y])))
            remaining_agents -= min(remaining_agents, math.floor(n/max_terminal*(100-stops[y])))
        else:
            temp[stops[y]] += min(remaining_agents, math.floor(n/max_terminal*(stops[y+1]-stops[y])/2))
            remaining_agents -= min(remaining_agents, math.floor(n/max_terminal*(stops[y+1]-stops[y])/2))
        y = y+1
    y = 0
    while remaining_agents > 0:
        temp[stops[y]] += 1
        remaining_agents -= 1
        y = (y+1)%m
    

    for i, t in enumerate(temp):
        if i > 0:
            temp[i] = t + temp[i-1]

    x = 1
    for i, s in enumerate(stops):
        if k - x +1 == m - i or temp[s] >= x*2*math.floor(n/k):
            solution.append(stops[i])
            x += 1
    return solution


def benchmark2(agents, stops, k):
    m = len(stops)
    n = len(agents)

    num_user = [0]*m
    for a in agents:
        num_user[closest_index(stops, a[0])] += 1
        num_user[closest_index(stops, a[1])] += 1

    return [stops[i] for i in sorted(range(m), key=lambda x: num_user[x])[-k:]]


def algorithm1(agents, stops, k):
    
    n = len(agents)
    m = len(stops)
    solution = [] # list of length k
    max_terminal = max (max(stops), max(a[1] for a in agents))
    temp = [0]*(max_terminal+1)
    for a in agents:
        temp[a[0]] += 1
        temp[a[1]] += 1

    

    for i, t in enumerate(temp):
        if i > 0:
            temp[i] = t + temp[i-1]

    x = 1
    for i, s in enumerate(stops):
        if k - x +1 == m - i or temp[s] >= x*2*math.floor(n/k):
            solution.append(stops[i])
            x += 1

    return solution




def alpha_experiment(r, number_of_runs):

    all_n = list(range(5, 26))
    all_m = list(range(5, 16))

    all_n.reverse()

    all_alpha = [0, 1, 2, 3, 4, 5, 6 7, 8, 9]
    
    start = time.time()

    for ten_alpha in all_alpha:
        alpha = ten_alpha/10

        ans = []
        for n in all_n:
            ans_n = []
            for m in all_m:
                ans_m = []
                for k in range(3, m):
                    core_cnt, pf_cnt = 0, 0
                    core_cnt2, pf_cnt2 = 0, 0
                    for i in range(number_of_runs):  ######################################
                        agents, stops = generate_random_instance(n, m)
                        sol = algorithm1(agents, stops, k)
                        core_flag, pf_flag = check_in_the_core(agents, stops, sol, alpha)
                        if core_flag == False:
                            core_cnt += 1
                        if pf_flag == False:
                            pf_cnt += 1
                        sol2 = benchmark(agents, stops, k)
                        core_flag2, pf_flag2 = check_in_the_core(agents, stops, sol2, alpha)
                        if core_flag2 == False:
                            core_cnt2 += 1
                        if pf_flag2 == False:
                            pf_cnt2 += 1
                    ans_m.append((core_cnt, pf_cnt, core_cnt2, pf_cnt2))
                ans_n.append(ans_m)

            ans.append(ans_n)

        file_name = 'Exp_{}_{}_runs_{}.json'.format(r, ten_alpha, number_of_runs)
        with open(file_name, 'w') as f:
            json.dump(ans, f)
        print("\t", alpha, "   ", (time.time() - start)/60, " minutes")

runs = [100, 100, 100, 100]

for i in range(len(runs)):
    alpha_experiment(i, runs[i])
