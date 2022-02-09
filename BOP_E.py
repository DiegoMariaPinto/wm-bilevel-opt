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
disdur = load_instance('disdur_' + instance_name + '.json')

NF = 10  # number of facilities
NC = 30  # number of clients
ND = 5   # number of depots
NV = 8   # number of vehicles
# corresponding sets:
F = list(range(NF))  # facility list
C = list(range(NF, NC + NF))
D = list(range(NC + NF, ND + NC + NF))
N = F + C + D
V = list(range(NV))

########################
# SP specific parameters
H = [1, 2, 3]  # 1 = S, 2 = M, 3 = L
CAPH = {1: 500, 2: 1000, 3: 1500}
EM = {1: 20, 2: 25, 3: 30}  # test different shapes
FCost = {1: 100, 2: 150, 3: 180}  # facility installation cost
capf = {(j, h): CAPH[h] for (j, h) in itertools.product(F, H)}
c = {(j, h): FCost[h] for (j, h) in itertools.product(F, H)}
B = 3000  # budjet for facilities opening  # 1600
em_f = {(j, h): EM[h] for (j, h) in itertools.product(F, H)}
NS = NF  # upper bound of the number of clusters
S = list(range(NS))
# sc is the safe capacity for each facility j of size h
sc = {(j, h): 0.3 * CAPH[h] for (j, h) in itertools.product(F, H)}

# SP_params = {'H':H, 'CAPH' = CAPH, etc..}
#########################
# OP specific parameters
t = {(a, b): disdur[(a, b)]['duration'] for (a, b) in itertools.product(N, N)}
truck_em_coeff = 1.2
em_t = {(a, b): truck_em_coeff * disdur[(a, b)]['distance'] for (a, b) in itertools.product(N, N)}
random_cv = np.random.randint(2500, 4500, NV).tolist()  # vehicles capacities
cv = {l: random_cv[l] for l in V}
random_T = np.random.randint(600, 900, NV).tolist()  # maximum servicing times per tour (electic or combustion) 3,8
T = {l: random_T[l] for l in V}
random_P = np.random.randint(50, 100, NF).tolist()  # maximum penalty for a facility
P = {j: random_P[j] for j in F}
a_matrix = {(k, l): 1 for (k, l) in itertools.product(D, V) if
            k - D[0] == l - V[0]}  # trucks distribution across the depots

for (k, l) in itertools.product(D, V):
    if k - D[0] != l - V[0]:
        a_matrix[(k, l)] = 0
a_matrix[(40,5)] = 1
a_matrix[(41,6)] = 1
a_matrix[(42,7)] = 1
# a_k_l == 1 if truck l start its tour from depot k

# OP_params = {....}
#########################

# demand vector d= ?
random_d = np.random.randint(100, 250, NC).tolist()  # demand of clients i in C
d = {i: random_d[i - NF] for i in C}

# gamma vector of gamma_1,2,3
params = []
gamma = {'1': 0.3, '2': 0.3, '3': 0.4}


def SP_model(params, gamma, OP_vars, gap_tol, time_limit, get_first_sol_SP=False, SP_vars_to_fix=None):
    m = Model('SP')

    # Variables
    if get_first_sol_SP:
        z = None
        x = {(i, s): m.addVar(vtype=GRB.BINARY, name='x({},{})'.format(i, s))
             for i in C for s in S}
        r = {(j, h, s): m.addVar(vtype=GRB.BINARY, name='r({},{},{})'.format(j, h, s))
             for j in F for h in H for s in S}
        y = {(j): m.addVar(vtype=GRB.BINARY, name='y({})'.format(j))
             for j in F}
        n = {(j, h, s): m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='n({},{},{})'.format(j,h,s))
             for j in F for h in H for s in S}
        # q = {(s) : m.addVar(vtype=GRB.BINARY, name='q({})'.format(s))
        #      for s in S}
    else:
        z = OP_vars['z']
        x = {(i, s): m.addVar(lb=SP_vars_to_fix['x'][i, s], ub=SP_vars_to_fix['x'][i, s], vtype=GRB.BINARY,
                              name='x({},{})'.format(i, s))
             for i in C for s in S}
        r = {(j, h, s): m.addVar(lb=SP_vars_to_fix['r'][j, h, s], ub=SP_vars_to_fix['r'][j, h, s], vtype=GRB.BINARY,
                                 name='r({},{},{})'.format(j, h, s))
             for j in F for h in H for s in S}
        y = {(j): m.addVar(lb=SP_vars_to_fix['y'][j], ub=SP_vars_to_fix['y'][j], vtype=GRB.BINARY,
                           name='y({})'.format(j))
             for j in F}
        n = {(j, h, s): m.addVar(lb=SP_vars_to_fix['n'][j, h, s], ub=SP_vars_to_fix['n'][j, h, s], vtype=GRB.CONTINUOUS,
                           name='n({},{},{})'.format(j,h,s))
             for j in F for h in H for s in S}
        # q = {(s) : m.addVar(SP_vars_to_fix['q'][s], ub=SP_vars_to_fix['q'][s], vtype=GRB.BINARY, name='q({})'.format(s))
        #      for s in S}

        # obj.fun. linearization vars

        g = {(a, b, l, s): m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='g({},{},{},{})'.format(a, b, l, s))
             for a in C for b in C for l in V for s in S}
        u = {(a, b, s): m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='u({},{},{})'.format(a, b, s))
             for a in C for b in C for s in S}

    # Constraints
    # (1)
    for i in C:
        m.addConstr(quicksum(x[i, s] for s in S) == 1, name='C1_client_to_cluster({})'.format(i))
    # (2)
    for j in F:
        m.addConstr(quicksum(r[j, h, s] for s in S for h in H) == y[j], name='C2_one_size_only({})'.format(j))

    if get_first_sol_SP:
        # (3)
        for s in S:
            m.addConstr(quicksum(x[i, s] * d[i] for i in C) <= quicksum(r[j, h, s] * capf[j, h] - (r[j, h, s] - n[j,h,s]) * sc[j, h] for j in F for h in H),
                        name='C3_demand_to_cluster({})'.format(s))
    # (4)
    for i in C:
        for s in S:
            m.addConstr(x[i, s] <= quicksum(r[j, h, s] for j in F for h in H),
                        name='C4_facility_to_cluster({},{})'.format(i, s))
    # (5) new
    for j in F:
        for h in H:
            for s in S:
                m.addConstr(n[j,h,s] <= r[j,h,s], name='C5_new_nj_iff_facility_is_open({},{},{})'.format(j,h,s))

    c = {(j, h): FCost[h] for (j, h) in itertools.product(F, H)}
    # (6)
    m.addConstr(quicksum(c[j, h] * r[j, h, s] for j in F for h in H for s in S) <= B, name='C6_budjet_constr')

    max_j_for_s = 2
    for s in S:
        m.addConstr(quicksum(r[j,h,s] for j in F for h in H) <= max_j_for_s, name='C_min_clusters')

    # # (7) constraint a minimum number of clusters
    # for s in S:
    #     m.addConstr(q[s] <= quicksum(x[i,s] for i in C), name='C7_min_cluster_1({})'.format(s))
    # # (8)
    # for s in S:
    #     m.addConstr(NC*q[s] >= quicksum(x[i,s] for i in C), name='C8_min_cluster_2({})'.format(s))
    # # (9)
    # min_cluster = 3
    # m.addConstr(quicksum(q[s] for s in S) >= min_cluster, name='C9_min_cluster_3')


    # linearization related constraints

    if not get_first_sol_SP:
        # (21): g var lb is included in var definition
        # (22)
        for a in C:
            for b in C:
                for l in V:
                    for s in S:
                        m.addConstr(g[a, b, l, s] >= z[l, a, b] - u[a, b, s])
        # (23)
        for a in C:
            for b in C:
                for s in S:
                    m.addConstr(u[a, b, s] >= x[a, s] + x[b, s] - 1)
        # (24)
        for a in C:
            for b in C:
                for s in S:
                    m.addConstr(u[a, b, s] <= (x[a, s] + x[b, s]) / 2)

    # objective function

    if get_first_sol_SP:
        m.setObjective(gamma['1'] * quicksum(r[j, h, s] * em_f[j, h] for j in F for h in H for s in S) +
                       gamma['2'] * quicksum(n[j,h,s] * P[j] for j in F for h in H for s in S))
    else:
        m.setObjective(gamma['1'] * quicksum(z[l, a, b] * em_t[a, b] for a in N for b in N for l in V) +
                       gamma['1'] * quicksum(r[j, h, s] * em_f[j, h] for j in F for h in H for s in S) +
                       gamma['2'] * quicksum(n[j, h, s] * P[j] for j in F for h in H for s in S) +
                       gamma['3'] * quicksum(g[a, b, l, s] for a in C for b in C for l in V for s in S))

    ################# solve the formulation ####################

    m.Params.MIPGap = gap_tol
    m.Params.TimeLimit = time_limit

    m.modelSense = GRB.MINIMIZE
    m.update()
    m.write('SP_model.lp')
    m.optimize()

    status = m.status
    if status == GRB.Status.INFEASIBLE:
        status = "infeasible"
        print(status)
        m.computeIIS()
        for c in m.getConstrs():
            if c.IISConstr:
                print(c.constrName)
        m.write('SP.ilp')
        return 'infeasible', 'infeasible'

    if status == GRB.Status.OPTIMAL or status == 9:  # 9 is equivalent to 'Time Limit Reached'
        print(status)
        optObjVal = m.getAttr(GRB.Attr.ObjVal)
        bestObjBound = m.getAttr(GRB.Attr.ObjBound)
        Runtime = m.Runtime
        Gap = m.MIPGap

        vars_opt = []
        x_opt_dict = {}
        r_opt_dict = {}
        y_opt_dict = {}
        n_opt_dict = {}
        q_opt_dict = {}
        for var in m.getVars():
            vars_opt.append([var.VarName, var.x])
            if var.VarName.startswith('x'):
                x_opt_dict[eval(var.VarName[2:-1])] = round(var.x)
            if var.VarName.startswith('r'):
                r_opt_dict[eval(var.VarName[2:-1])] = round(var.x)
            if var.VarName.startswith('y'):
                y_opt_dict[eval(var.VarName[2:-1])] = round(var.x)
            if var.VarName.startswith('n'):
                n_opt_dict[eval(var.VarName[2:-1])] = var.x
            if var.VarName.startswith('q'):
                q_opt_dict[eval(var.VarName[2:-1])] = round(var.x)

        vars_opt = pd.DataFrame.from_records(vars_opt, columns=["variable", "value"])
        vars_opt.to_excel('risultati_SP.xlsx')

        x_opt = vars_opt[vars_opt['variable'].str.contains("x", na=False)]
        r_opt = vars_opt[vars_opt['variable'].str.contains("r", na=False)]
        y_opt = vars_opt[vars_opt['variable'].str.contains("y", na=False)]
        n_opt = vars_opt[vars_opt['variable'].str.contains("n", na=False)]
        q_opt = vars_opt[vars_opt['variable'].str.contains("q", na=False)]

        x_opt['value'].apply(pd.to_numeric)
        r_opt['value'].apply(pd.to_numeric)
        y_opt['value'].apply(pd.to_numeric)
        n_opt['value'].apply(pd.to_numeric)
        q_opt['value'].apply(pd.to_numeric)

        df_vars_list = [x_opt, r_opt, y_opt, n_opt, q_opt]

        opt_vars = {'x': x_opt_dict, 'r': r_opt_dict, 'y': y_opt_dict, 'n': n_opt_dict, 'q': q_opt_dict}

        return opt_vars, optObjVal, df_vars_list


def OP_model(params, SP_vars, gap_tol, time_limit):
    m = Model('SP')

    y = SP_vars['y']
    r = SP_vars['r']

    # OP Variables

    h = {(l, a): m.addVar(vtype=GRB.BINARY, name='h({},{})'.format(l, a))
         for l in V for a in N}

    z = {(l, a, b): m.addVar(vtype=GRB.BINARY, name='z({},{},{})'.format(l, a, b))
         for l in V for a in N for b in N}

    e = {(l,a): m.addVar(lb=0, vtype=GRB.INTEGER, name='e({},{})'.format(l,a))
         for l in V for a in N}

    v = {(l,j): m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='v({},{})'.format(l,j))
         for l in V for j in F}

    # obj.fun. linearization var
    w = m.addVar(vtype=GRB.CONTINUOUS, obj=1, name='w')

    # Constraints
    # (8)
    for l in V:
         m.addConstr(quicksum(h[l, j] for j in F) == quicksum(z[l, k, i] for k in D for i in C),
                     name='C_8_({})'.format(l))
    # (9)
    for i in C:
        m.addConstr(quicksum(h[l, i] for l in V) == 1, name='C_9_({})'.format(i))

    # (10)
    for l in V:
        m.addConstr(quicksum(h[l, k] for k in D) == 1, name='C_10_({})'.format(l))
    # (10) bis
    for l in V:
        m.addConstr(quicksum(h[l, j] for j in F) == 1, name='C_10_bis_({})'.format(l))
    # (11)
    for l in V:
        m.addConstr(quicksum(h[l, i] * d[i] for i in C) <= cv[l], name='C_11_({})'.format(l))
    # 12)
    for k in D:
        for l in V:
            m.addConstr(quicksum(z[l, k, i] for i in C) <= a_matrix[k, l], name='C_12_({},{})'.format(k, l))
    # 12) BIS
    for k in D:
        for l in V:
            m.addConstr(quicksum(z[l, j, k] for j in F) <= a_matrix[k, l], name='C_12_BIS({},{})'.format(k, l))
    # # 13)
    # for j in F:
    #     for l in V:
    #         m.addConstr(quicksum(z[l, i, j] for i in C) <= y[j], name='C_13_({},{})'.format(j, l))

    # 14)
    for j in F:
        for l in V:
            m.addConstr(quicksum(z[l, j, i] for i in C) == 0, name='C_14_({},{})'.format(j, l))
    # 15)
    for j in F:
        for l in V:
            m.addConstr(quicksum(z[l, i, j] for i in C) <= h[l, j], name='C_15_({},{})'.format(j, l))
    # 16)
    for j in F:
        for l in V:
            m.addConstr(quicksum(z[l, j, k] for k in D) == quicksum(z[l, i, j] for i in C),
                        name='C_16_({},{})'.format(j, l))
    # # 17)
    # for i in C:
    #     for l in V:
    #         m.addConstr(quicksum(z[l, i_1, i] for i_1 in C + D if i != i_1) == quicksum(
    #             z[l, i, i_1] for i_1 in C if i != i_1) + quicksum(z[l, i, j] for j in F),
    #                     name='C_17_({},{})'.format(i, l))
    # 18)
    for i in C:
        for l in V:
            m.addConstr(quicksum(z[l, i, i_1] for i_1 in C if i != i_1) <= 1, name='C_18_({},{})'.format(i, l))
    # 19)
    for l in V:
        m.addConstr(quicksum(t[a, b] * z[l, a, b] for a in N for b in N if a != b) <= T[l], name='C_19_({})'.format(l))

    # 20) no loop over same node constraint:
    # for l in V:
    #     for i in N:
    #         m.addConstr(z[l, i, i] == 0, name='C_20_no_loop({},{})'.format(l, i))

    # New constraints by Pizzari and Pinto in 23/12/21 binding z and h to behave
    # 21)
    for l in V:
        for a in C:
            m.addConstr(quicksum(z[l,a,b] for b in C+F) == h[l,a], name='C_21_fromclient_to_clientORfacility_({},{})'.format(l,a))
    # 22)
    for l in V:
        for a in C:
            m.addConstr(quicksum(z[l,b,a] for b in D+C) == h[l,a], name='C_22_toclient_from_clientORdeposit_({},{})'.format(l,a))

    ############################################################################################
    ### no loop constraints:
    for l in V:
        for a in D+C:
            for b in C+F:
                m.addConstr((z[l,a,b]==1)>>(e[l,b]==e[l,a]+1), name='ordine({},{})'.format(l,a,b))

    for l in V:
        for a in D:
            m.addConstr(e[l,a]==0, name='ordine_dep({},{})'.format(l,a))

    for l in V:
        for a in N:
            m.addConstr(e[l,a]<=NC*h[l,a], name='controllo_ordine({},{})'.format(l,a))
    ############################################################################################

    for l in V:
        for j in F:
            m.addConstr((h[l, j] == 1) >> (quicksum(h[(l, i)]*d[i] for i in C) == v[l,j]), name='load_of_l_to_j({},{})'.format(l,j))

    for j in F:
            m.addConstr(quicksum(v[l,j] for l in V) <= quicksum(r[j,h,s]*capf[j,h] for h in H for s in S), name='load_of_j({})'.format(j))

    for l in V:
        for j in F:
            m.addConstr((h[l, j] == 0) >> (v[l, j] == 0),  name='load_of_l_not_to_j({},{})'.format(l, j))

    # 23)
    # for l in V:
    #     for a in D:
    #         m.addConstr(quicksum(z[l,a,b] for b in C)   == h[l,a], name='C_23_fromdepot_to_client_({},{})'.format(l,a))
    # linearization related constraints
    # (27)
    # for k in D:
    #     for k_1 in D:
    #         if k_1 != k:
    #             m.addConstr(
    #                 w >= quicksum(z[l, a, b] * t[a, b] for a in N for b in N for l in V if a_matrix[k, l] == 1) -
    #                 quicksum(z[l, a, b] * t[a, b] for a in N for b in N for l in V if a_matrix[k_1, l] == 1)
    #                 , name='C_27_({},{})'.format(k, k_1))

    # NEW linearization related constraints
    for k in D:
            m.addConstr(
                w >= quicksum(z[l, a, b] * t[a, b] for a in N for b in N for l in V if a_matrix[k,l] == 1)
                , name='C_27_({})'.format(k))

    ################# solve the formulation ####################

    m.Params.MIPGap = gap_tol
    m.Params.TimeLimit = time_limit

    m.Params.IntFeasTol = 1e-3 # default is 1e-5

    m.modelSense = GRB.MINIMIZE
    m.update()
    m.write('OP_model.lp')
    #########################################################
    # m = m.relax() #########################################
    #########################################################

    m.optimize()

    status = m.status
    if status == GRB.Status.INFEASIBLE:
        status = "infeasible"
        print(status)
        m.computeIIS()
        for c in m.getConstrs():
            if c.IISConstr:
                print(c.constrName)
        m.write('OP.ilp')
        return 'infeasible'

    if status == GRB.Status.OPTIMAL or status == 9:  # 9 is equivalent to 'Time Limit Reached'
        print(status)
        optObjVal = m.getAttr(GRB.Attr.ObjVal)
        bestObjBound = m.getAttr(GRB.Attr.ObjBound)
        Runtime = m.Runtime
        # Gap = m.MIPGap

        vars_opt = []
        h_opt_dict = {}
        z_opt_dict = {}
        e_opt_dict = {}
        v_opt_dict = {}

        for var in m.getVars():
            vars_opt.append([var.VarName, var.x])
            if var.VarName.startswith('h'):
                h_opt_dict[eval(var.VarName[2:-1])] = round(var.x)
            if var.VarName.startswith('z'):
                z_opt_dict[eval(var.VarName[2:-1])] = round(var.x)
            if var.VarName.startswith('e'):
                e_opt_dict[eval(var.VarName[2:-1])] = round(var.x)
            if var.VarName.startswith('v'):
                v_opt_dict[eval(var.VarName[2:-1])] = round(var.x)

        vars_opt = pd.DataFrame.from_records(vars_opt, columns=["variable", "value"])
        vars_opt.to_excel('risultati_SP.xlsx')

        h_opt = vars_opt[vars_opt['variable'].str.contains("h", na=False)]
        z_opt = vars_opt[vars_opt['variable'].str.contains("z", na=False)]
        e_opt = vars_opt[vars_opt['variable'].str.contains("e", na=False)]
        v_opt = vars_opt[vars_opt['variable'].str.contains("v", na=False)]

        h_opt['value'].apply(pd.to_numeric)
        z_opt['value'].apply(pd.to_numeric)
        e_opt['value'].apply(pd.to_numeric)
        v_opt['value'].apply(pd.to_numeric)

        df_vars_list = [h_opt, z_opt, e_opt, v_opt]

        opt_vars = {'h': h_opt_dict, 'z': z_opt_dict, 'e': e_opt_dict, 'v': v_opt_dict}

        return opt_vars, df_vars_list



def get_facility_load(OP_opt_vars, SP_opt_vars):

    h = OP_opt_vars['h']
    y = SP_opt_vars['y']

    loads = {}
    for l in V:
        load = 0
        for i in C:
            load += h[(l, i)] * d[i]
        loads[l] = load

    j_load = {}
    for j in F:
        if y[j] == 1:
            load_j = 0
            for l in V:
                load_j += h[l, j] * loads[l]

            j_load[j] = load_j

    return j_load

def get_cluster_load(SP_opt_vars):

    x = SP_opt_vars['x']
    s_load = {}
    for s in S:
        demand_s = 0
        for i in C:
            demand_s += x[i, s] * d[i]
        s_load[s] = demand_s

    return s_load


def get_closest_facility_list(j_0):
    distances = {}
    for j in F:
        if j != j_0:
            distances[j] = dist[(j_0, j)]

    closest_facility_list = [(j, dist_value) for j, dist_value in
                             distances.items()]  # list of all facilities j and their distance from j_0 input
    closest_facility_list.sort(key=lambda x: x[1], reverse=False)  # list is sort in descending order w.r.t n_j

    return closest_facility_list


def evalute_SP_objval(OP_opt_vars,SP_opt_vars):

    s_load = get_cluster_load(SP_opt_vars_init)
    # evaluate load of open facility due to trucks routing
    j_load = get_facility_load(OP_opt_vars, SP_opt_vars)
    # update and fix var. n according to actual j_load
    r = SP_opt_vars['r']
    open_list = [r_var for r_var, value in r.items() if value == 1]

    for r_jhs in open_list:
        j = r_jhs[0]
        h = r_jhs[1]
        s = r_jhs[2]
        free_capf = capf[(j, h)] - j_load[j]
        if free_capf >= sc[(j, h)]:
            SP_opt_vars['n'][j, h, s] = 0
        elif free_capf >= 0 and free_capf < sc[(j, h)]:
            SP_opt_vars['n'][j, h, s] = (sc[(j, h)] - free_capf) / sc[(j, h)]

            # [(n_var, val) for n_var, val in n.items() if val > 0]

    _, SP_optval, __ = SP_model(params, gamma, OP_opt_vars, gap_tol, time_limit_SP, False, SP_opt_vars)

    return SP_optval

def find_size_and_cluster_of_j(j, H, S, list):
    for s in S:
        for h in H:
            if (j, h, s) in list:
                return h , s
    return


def get_j_usage(open_list, j_load, capf):
    j_usage = {}
    for jhs in open_list:
        j = jhs[0]
        h = jhs[1]
        j_usage[j] = j_load[j] / capf[(j, h)]

    return j_usage

if __name__ == '__main__':

    time_limit_SP = 20
    time_limit_OP = 120
    gap_tol = 0.0025  # 1e-5

    OP_opt_vars = None

    print('########################### \n FIRST ATTEMPT TO SOLVE SP \n########################### ')
    get_first_sol_SP = True
    SP_opt_vars_init, _, SP_vars_init_list = SP_model(params, gamma, OP_opt_vars, gap_tol, time_limit_SP,
                                                                    get_first_sol_SP)
    get_first_sol_SP = False

    # evaluate load of clusters due to clients assignement
    s_load = get_cluster_load(SP_opt_vars_init)

    print('########################### \n FIRST ATTEMPT TO SOLVE OP \n###########################')
    OP_opt_vars, OP_vars_list = OP_model(params, SP_opt_vars_init, gap_tol, time_limit_OP)

    # evaluate load of open facility due to trucks routing
    j_load = get_facility_load(OP_opt_vars, SP_opt_vars_init)

    # update var. n according to actual j_load
    r = SP_opt_vars_init['r']
    open_list = [r_var for r_var, value in r.items() if value == 1]

    for r_jhs in open_list:
        j = r_jhs[0]
        h = r_jhs[1]
        s = r_jhs[2]
        free_capf = capf[(j,h)] - j_load[j]
        if free_capf >= sc[(j,h)]:
            SP_opt_vars_init['n'][j,h,s] = 0
        elif free_capf >= 0 and free_capf < sc[(j,h)]:
            SP_opt_vars_init['n'][j,h,s] = (sc[(j,h)] - free_capf)/sc[(j,h)]


    print('########################### \n First Evaluation of total SP objective value \n###########################')
    SP_opt_vars, SP_optval, SP_vars_list = SP_model(params, gamma, OP_opt_vars, gap_tol, time_limit_SP, get_first_sol_SP,
                                         SP_opt_vars_init)

    SP_obj_evolution = {}
    SP_obj_evolution[0] = SP_optval
    # FOR PUSH
    Heuristic = True
    if Heuristic:
        print('####### Heuristic START HERE #######')
        max_k = 12
        for count in range(1,max_k):

            print('Iteration n. '+str(count))

            # Heuristic iteration start:
            x = SP_opt_vars['x']
            y = SP_opt_vars['y']
            n = SP_opt_vars['n']
            r = SP_opt_vars['r']

            B_used = sum(c[j, h] * r[j, h, s] for j in F for h in H for s in S)

            dist = {(a, b): disdur[(a, b)]['duration'] for (a, b) in itertools.product(N, N)}

            open_list = [r_var for r_var, value in r.items() if value == 1]  # list of all (j,h,s) s.t. r[j,h,s] = 1

            ###################################################################
            ## 1 step: enlarge size of open facilities using safety stock ##
            if max(n.values()) > 0:  # at least one facility is using safety stock --> leader cost to decrease
                ss_used_list = [(n_var, value) for n_var, value in n.items() if
                                value > 0]  # list of all facilities j with n_j > 0
                ss_used_list.sort(key=lambda x: x[1], reverse=True)  # list is sort in descending order w.r.t n_j
                for elem in ss_used_list:  # for each of these facilities where safaty stock is used (in descending oreder w.r.t n_j)
                    j = elem[0][0]
                    h = elem[0][1]
                    s = elem[0][2]
                    if h != 3 and B_used + c[j, h + 1] - c[j, h] <= B:  # upgrade its size if possible and check if budjet is available
                        r[j, h, s] = 0
                        r[j, h + 1, s] = 1
                        B_used += c[j, h + 1] - c[j, h]
                        print('facility j = ' + str(j) + ' has been enlarged to size ' + str(h + 1))

            ############################################################
            ## 2 step: attempt to modify Y by closing or opening some facility ##
            ## 2.1 step: compute capacity utilization of each open facility according to follower behaviour Z
            # compute percentage of stock usage w.r.t. facility capacity
            j_load = get_facility_load(OP_opt_vars, SP_opt_vars)
            j_usage = get_j_usage(open_list, j_load, capf)
            if min(j_usage.values()) < 0.4:
                j_to_close = min(j_usage, key=j_load.get)
                print('found facility to close: is facility ' + str(j_to_close) + ' being used at '+str(min(j_usage.values())))
                y[j_to_close] = 0
                h_to_close, s_to_close = find_size_and_cluster_of_j(j_to_close, H, S, open_list)
                r[j_to_close, h_to_close, s_to_close] = 0

                # open the facility closest to j
                j_to_help = max(n, key=n.get)[0]
                closest_facility_list = get_closest_facility_list(j_to_help)
                for j in closest_facility_list:
                    if j[0] not in [elem[0] for elem in open_list]:
                        j_to_open = j[0]
                        break

                y[j_to_open] = 1
                r[j_to_open, h_to_close, s_to_close] = 1
                print('found facility to open: is facility ' + str(j_to_open))
            # New Y and R vars are give to OP for a new solution:
            SP_opt_vars['y'] = y
            SP_opt_vars['r'] = r
            print('########################### \n' + str(count) + 'th iteration of SOLVING OP \n###########################')
            OP_opt_vars, OP_vars_list = OP_model(params, SP_opt_vars, gap_tol, time_limit_OP)

            SP_optval_k = evalute_SP_objval(OP_opt_vars,SP_opt_vars)
            SP_obj_evolution[count] = SP_optval_k

            # # cluster definition may be inefficient --> attempt to change facility distribution across clusters
            # else:
            #     print('ciao')


