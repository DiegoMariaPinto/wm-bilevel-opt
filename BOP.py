import itertools
from gurobipy import *
from gurobipy import quicksum
from gurobipy import GRB
from gurobipy import Model

import pandas as pd
import numpy as np
from Get_instances import load_instance
np.random.seed(0)

# Instance Load:
instance_name = 'inst_#1'
disdur = load_instance('disdur_'+instance_name+'.json')

NF = 10 # number of facilities
NC = 30 # number of clients
ND = 5  # number of depots
NV = 8  # number of vehicles
# corresponding sets:
F = list(range(NF))
C = list(range(NF,NC+NF))
D = list(range(NC+NF, ND+NC+NF))
N = F + C + D
V = list(range(NV))

########################
# SP specific parameters
H = ['S', 'M', 'L']
CAPH = {'S':500, 'M':1000, 'L':1500}
EM = {'S':20, 'M':25, 'L':30} # test different shapes
cf  = {(j,h): CAPH[h] for (j, h) in itertools.product(F,H)}
c_cost = {(j,h): 1 for (j, h) in itertools.product(F,H)}
em = {(j,h):EM[h] for (j,h) in itertools.product(F,H)}
NS = NF # upper bound of the number of clusters
S = list(range(1,NS+1))
# sc is the safe capacity for each facility j of size h
sc = {(j,h): 0.3*CAPH[h] for (j,h) in itertools.product(F,H)}
#########################
# OP specific parameters
t  = {(a,b):disdur[(a,b)]['distance'] for (a,b) in itertools.product(N,N)}
random_cv = np.random.randint(30,60,NV).tolist() # vehicles capacities
cv = {l: random_cv[l] for l in V}
random_T = np.random.randint(3,8,NV).tolist() # maximum servicing times per tour (electic or combustion)
T = {l: random_T[l] for l in V}
random_P = np.random.randint(50,100,NF).tolist() # maximum penalty for a facility
P = {j: random_P[j] for j in F}
a_matrix = {(k,l) : 1 for (k,l) in itertools.product(D,V)} # trucks distribution across the depots
#########################

# demand vector d= ?
d = {} # ?
# gamma vector of gamma_1,2,3
params = []
gamma = {'1':0.3,'2':0.3,'3':0.4}

def SP_model(params, gamma, OP_vars,
             solver, gap, time_limit):

    m = Model('SP')

    z = OP_vars['z']

    # Variables

    x = {(i, s): m.addVar(lb=0, ub=1, vtype=GRB.BINARY, obj= None, name='x({},{})'.format(i, s))
         for i in C for s in S}

    r = {(j, h, s): m.addVar(lb=0, ub=1, vtype=GRB.BINARY, obj= None, name='r({},{},{})'.format(j, h, s))
         for j in F for h in H, for s in S}

    y = {(j): m.addVar(lb=0, ub=1, vtype=GRB.BINARY, obj= None, name='L({})'.format(j))
         for j in F}

    n = {(j): m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, obj=None, name='n({})'.format(j))
         for j in F}

    # obj.fun. linearization vars

    g = {(a,b,l,s): m.addVar(lb=0, vtype=GRB.CONTINUOUS, obj='None', name ='g({},{},{},{})'.format(a,b,l,s))
         for a in N for b in N for l in V for s in S}

    u = {(a,b,s): m.addVar(lb=0,ub=1, vtype=GRB.BINARY, obj='None', name = 'u({},{},{})'.format(a,b,s))
         for a in N for b in N for s in S}

    # Constraints
    # (1)
    for i in C:
        m.addConstr(quicksum(x[i,s] for s in S) == 1, name='client_to_cluster({})'.format(i))
    # (2)
    for j in F:
        m.addConstr(quicksum(r[j,h,s] for s in S for h in H) <= y[j], name='one_size_only({})'.format(j))
    # (3)
    for s in S:
        m.addConstr(quicksum(x[i,s]*d[i] for i in C) <= quicksum(r[j,h,s]*cf[j,h] - (r[j,h,s]-n[j])*sc[j,h]  for j in F for h in H))
    # (4)
    for i in C:
        for s in S:
            m.addConstr(x[i,s] <= quicksum(r[j,h,s] for j in F for h in H))
    # (5)
    for s in S:
        m.addConstr(quicksum(x[i, s] for i in C) <= quicksum(r[j, h, s] for j in F for h in H))

    # linearization related constraints

    # (21): g var lb is included in var definition
    # (22)
    for a in N:
        for b in N:
            for l in V:
                for s in S:
                    m.addConstr(g[a,b,l,s] >= z[l,a,b] - u[a,b,s])
    # (23)
    for a in N:
        for b in N:
            for s in S:
                m.addConstr(u[a,b,s] >= x[a,s] + x[b,s] - 1)
    # (24)
    for a in N:
        for b in N:
            for s in S:
                m.addConstr(u[a,b,s] <= (x[a,s] + x[b,s])/2)

    # objective function

    m.setObjective(gamma['1']*quicksum(z[l,a,b]*em[a,b] for a in N for b in N for l in V) +
                   gamma['2']*quicksum(n[j]*P[j] for j in F) +
                   gamma['3']*quicksum(g[a,b,l,s] for a in N for b in N for l in V for s in S))

    ################# solve the formulation ####################

    m.setParam(GRB.Param.TimeLimit, 120)
    m.setParam(GRB.Params.MIPGap,  0.05)

    m.modelSense = GRB.MINIMIZE
    m.update()
    m.optimize()

    status = m.status
    if status == GRB.Status.INFEASIBLE:
        status = "infeasible"
        print(status)
        m.computeIIS()
        m.write('VRP.ilp')
        return 'infeasible'

    if status == GRB.Status.OPTIMAL or status == 9: # 9 is equivalent to 'Time Limit Reached'
        print(status)
        optObjVal = m.getAttr(GRB.Attr.ObjVal)
        bestObjBound = m.getAttr(GRB.Attr.ObjBound)
        Runtime = m.Runtime
        Gap = m.MIPGap

        x = pd.Series(x).apply(lambda x: x.X)
        r = pd.Series(T).apply(lambda x: x.X)
        y = pd.Series(L).apply(lambda x: x.X)
        n = pd.Series(n).apply(lambda x: x.X)

        vars_opt = []
        for v in m.vars:
            vars_opt.append([v.name, v.x])

        vars_opt = pd.DataFrame.from_records(vars_opt, columns=["variable", "value"])

        x_opt = vars_opt[vars_opt['variable'].str.contains("x", na=False)]
        r_opt = vars_opt[vars_opt['variable'].str.contains("r", na=False)]
        y_opt = vars_opt[vars_opt['variable'].str.contains("y", na=False)]
        n_opt = vars_opt[vars_opt['variable'].str.contains("n", na=False)]

        x_opt['value'] = x_opt['value'].apply(pd.to_numeric).astype(int)
        r_opt['value'] = r_opt['value'].apply(pd.to_numeric).astype(int)
        y_opt['value'] = y_opt['value'].apply(pd.to_numeric).astype(int)
        n_opt['value'] = n_opt['value'].apply(pd.to_numeric).astype(int)

        opt_vars = {'x': x_opt, 'r': r_opt, 'y': y_opt, 'n': n_opt}
        
        return opt_vars


def OP_model(params, SP_vars, solver, gap, time_limit):
    m = Model('SP')

    y = SP_vars['y']

    # OP Variables

    h = {(l, a): m.addVar(lb=0, ub=1, vtype=GRB.BINARY, obj=None, name='h({},{})'.format(l, a))
         for l in V for a in N}

    z = {(l,a,b): m.addVar(lb=0, ub=1, vtype=GRB.BINARY, obj=None, name='z({},{},{})'.format(l, a, b))
         for l in V for a in N for b in N}

    # obj.fun. linearization var

    w = m.addVar(vtype=GRB.CONTINUOUS, obj=1, name = 'w')

    # Constraints
    # (7)
    for k in D:
        for l in V:
            m.addConstr(quicksum(h[l, j] for j in F) == quicksum(z[l,k,i] for k in D for i in C))
    # (8)
    for i in C:
        m.addConstr(quicksum(h[l,i] for l in V) == 1)
    # (9)
    for l in V:
        m.addConstr(quicksum(h[l,k] for k in D) == 1)
    # (10)
    for l in V:
        m.addConstr(quicksum(h[l,i]*d[i] for i in C) <= cv[l])
    # 11)
    for k in D:
        for l in V:
            m.addConstr(quicksum(z[l,k,i] for i in C) <= a_matrix[k,l])
    # 12)
    for j in F:
        for l in V:
            m.addConstr(quicksum(z[l,i,j] for i in C) <= y[j])
    # 13)
    for j in F:
        for l in V:
            m.addConstr(quicksum(z[l,j,i] for i in C) == 0)
    # 14)
    for j in F:
        for l in V:
            m.addConstr(quicksum(z[l,i,j] for i in C) <= h[l,j])
    # 15)
    for j in F:
        for l in V:
            m.addConstr(quicksum(z[l,j,k] for k in D) == quicksum(z[l,i,j] for i in C))
    # 16)
    for i in C:
        for l in V:
            m.addConstr(quicksum(z[l,i_1,i] for i_1 in C+D if i != i_1) == quicksum(z[l,i,i_1] for i_1 in C if i != i_1) + quicksum(z[l,i,j] for j in F))
    # 17)
    for i in C:
        for l in V:
            m.addConstr(quicksum(z[l,i,i_1] for i_1 in C if i != i_1) <= 1)
    # 18)
    for l in V:
        m.addConstr(quicksum(t[a,b]*z[l,a,b] for a in N for b in N if a != b) <= T[l])

    # linearization related constraints
    # (27)
    for k in D:
        for k_1 in D:
            if k_1 != k:
                m.addConstr( w >= quicksum(z[l,a,b]*t[a,b] for a in N for b in N for l in V if a[k,l]   == 1) -
                                  quicksum(z[l,a,b]*t[a,b] for a in N for b in N for l in V if a[k_1,l] == 1))

    ################# solve the formulation ####################

    m.setParam(GRB.Param.TimeLimit, 120)
    m.setParam(GRB.Params.MIPGap, 0.05)

    m.modelSense = GRB.MINIMIZE
    m.update()
    m.optimize()

    status = m.status
    if status == GRB.Status.INFEASIBLE:
        status = "infeasible"
        print(status)
        m.computeIIS()
        m.write('VRP.ilp')
        return 'infeasible'

    if status == GRB.Status.OPTIMAL or status == 9:  # 9 is equivalent to 'Time Limit Reached'
        print(status)
        optObjVal = m.getAttr(GRB.Attr.ObjVal)
        bestObjBound = m.getAttr(GRB.Attr.ObjBound)
        Runtime = m.Runtime
        Gap = m.MIPGap

        h = pd.Series(h).apply(lambda x: x.X)
        z = pd.Series(z).apply(lambda x: x.X)

        vars_opt = []
        for v in m.vars:
            vars_opt.append([v.name, v.x])

        vars_opt = pd.DataFrame.from_records(vars_opt, columns=["variable", "value"])

        h_opt = vars_opt[vars_opt['variable'].str.contains("h", na=False)]
        z_opt = vars_opt[vars_opt['variable'].str.contains("z", na=False)]

        h_opt['value'] = h_opt['value'].apply(pd.to_numeric).astype(int)
        z_opt['value'] = z_opt['value'].apply(pd.to_numeric).astype(int)


        opt_vars = {'h': h_opt, 'z': z_opt}

        return opt_vars

