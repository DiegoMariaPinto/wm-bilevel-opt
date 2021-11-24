import itertools
from gurobipy import *
from gurobipy import quicksum
from gurobipy import GRB
from gurobipy import Model

import pandas as pd
import numpy as np
np.random.seed(0)

# Dataset Load
network_data = pd.read_excel('WM_network_instance.xlsx')

def create_instance(network_data, NF,NC,ND):



    # inst_dist is a dictionary of distances between all instance nodes
    # inst_time is a dictionary of travel time between all instance nodes
    return inst_dist, inst_time


H = ['S', 'M', 'L']
CAPH = {'S':500, 'M':1000, 'L':1500}
EM = {'S':20, 'M':25, 'L':30} # test different shapes
NF = 10 # number of facilities
NC = 10 # number of clients
ND = 5  # number of depots
NV = 8  # number of vehicles
# corresponding sets:
F = list(range(1,       NF+1))
C = list(range(NF+1,    NC+NF+1))
D = list(range(NC+NF+1, ND+NC+NF+1))
N = F + C + D
V = list(range(1,NV+1))

c_cap  = {(j,h): CAPH[h] for (j, h) in itertools.product(F,H)}
c_cost = {(j,h): 1 for (j, h) in itertools.product(F,H)}
em = {(j,h):EM[h] for (j,h) in itertools.product(F,H)}
t  = {(a,b): inst_dist[(a,b)] for (a,b) in itertools.product(N,N)}

########################
# SP specific parameters
NS = 6
S = list(range(1,NS+1))
sc = {(j,h): 0.3*CAPH[h] for (j,h) in itertools.product(F,H)}
#########################
# OP specific parameters

random_cv = np.random.randint(30,60,NV).tolist() # vehicles capacities
cv = {l: random_cv[l] for l in V}
random_T = np.random.randint(3,8,NV).tolist() # maximum servicing times per tour (electic or combustion)
T = {l: random_T[l] for l in V}
#########################
params = []

def OP_model(params, solver, gap, time_limit):

    m = Model('OP')