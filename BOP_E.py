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
ND = 5  # number of depots
NV = 8  # number of vehicles
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
M = 50
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


def SP_model(params, gamma, OP_vars, gap_tol, time_limit, get_single_sol_SP=False, SP_vars_to_fix=None):
    m = Model('SP')

    # Variables
    if get_single_sol_SP:
        z = None
        x = {(i, s): m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x({},{})'.format(i, s))
             for i in C for s in S}
        r = {(j, h, s): m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='r({},{},{})'.format(j, h, s))
             for j in F for h in H for s in S}
        y = {(j): m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y({})'.format(j))
             for j in F}
        n = {(j,s): m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='n({},{})'.format(j,s))
             for j in F for s in S}
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
        n = {(j,s): m.addVar(lb=SP_vars_to_fix['n'][j], ub=SP_vars_to_fix['n'][j], vtype=GRB.CONTINUOUS,
                           name='n({},{})'.format(j,s))
             for j in F for s in S}

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
    # (3)
    for s in S:
        m.addConstr(quicksum(x[i, s] * d[i] for i in C) <= quicksum(r[j, h, s] * capf[j, h] - (r[j, h, s] - n[j,s]) * sc[j, h] for j in F for h in H),
                    name='C3_demand_to_cluster({})'.format(s))
    # (4)
    for i in C:
        for s in S:
            m.addConstr(x[i, s] <= quicksum(r[j, h, s] for j in F for h in H),
                        name='C4_facility_to_cluster({},{})'.format(i, s))
    # (5) new
    for j in F:
        for s in S:
            m.addConstr(n[j,s] <= quicksum(r[j,h,s] for h in H), name='C5_new_nj_iff_facility_is_open({},{})'.format(j,s))
    # (5) old
    # for s in S:
    #     m.addConstr(quicksum(x[i, s] for i in C) <= quicksum(r[j, h, s] for j in F for h in H), name='C5_empty_cluster({})'.format(s))

    c = {(j, h): FCost[h] for (j, h) in itertools.product(F, H)}
    # (6)
    m.addConstr(quicksum(c[j, h] * r[j, h, s] for j in F for h in H for s in S) <= B, name='C6_budjet_constr')

    # linearization related constraints

    if not get_single_sol_SP:
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

    if get_single_sol_SP:
        m.setObjective(gamma['1'] * quicksum(r[j, h, s] * em_f[j, h] for j in F for h in H for s in S) +
                       gamma['2'] * quicksum(n[j,s] * P[j] for j in F for s in S))
        # + quicksum(c[j, h] * r[j, h, s] for j in F for h in H for s in S))
    else:
        m.setObjective(gamma['1'] * quicksum(z[l, a, b] * em_t[a, b] for a in N for b in N for l in V) +
                       gamma['1'] * quicksum(r[j, h, s] * em_f[j, h] for j in F for h in H for s in S) +
                       gamma['2'] * quicksum(n[j,s] * P[j] for j in F for s in S) +
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
        for var in m.getVars():
            vars_opt.append([var.VarName, var.x])
            if var.VarName.startswith('x'):
                x_opt_dict[eval(var.VarName[2:-1])] = var.x
            if var.VarName.startswith('r'):
                r_opt_dict[eval(var.VarName[2:-1])] = var.x
            if var.VarName.startswith('y'):
                y_opt_dict[eval(var.VarName[2:-1])] = var.x
            if var.VarName.startswith('n'):
                n_opt_dict[eval(var.VarName[2:-1])] = var.x

        vars_opt = pd.DataFrame.from_records(vars_opt, columns=["variable", "value"])
        vars_opt.to_excel('risultati_SP.xlsx')

        x_opt = vars_opt[vars_opt['variable'].str.contains("x", na=False)]
        r_opt = vars_opt[vars_opt['variable'].str.contains("r", na=False)]
        y_opt = vars_opt[vars_opt['variable'].str.contains("y", na=False)]
        n_opt = vars_opt[vars_opt['variable'].str.contains("n", na=False)]

        x_opt['value'].apply(pd.to_numeric).astype(int)
        r_opt['value'].apply(pd.to_numeric).astype(int)
        y_opt['value'].apply(pd.to_numeric).astype(int)
        n_opt['value'].apply(pd.to_numeric).astype(int)

        df_vars_list = [x_opt, r_opt, y_opt, n_opt]

        opt_vars = {'x': x_opt_dict, 'r': r_opt_dict, 'y': y_opt_dict, 'n': n_opt_dict}

        return opt_vars, optObjVal, df_vars_list


def OP_model(params, SP_vars, gap_tol, time_limit, first_try, y_0=None):
    m = Model('SP')

    if first_try == True:
        y = y_0
    else:
        y = SP_vars['y']

    # OP Variables

    h = {(l, a): m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='h({},{})'.format(l, a))
         for l in V for a in N}

    z = {(l, a, b): m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z({},{},{})'.format(l, a, b))
         for l in V for a in N for b in N}

    e = {(l,a): m.addVar(lb=0, vtype=GRB.INTEGER, name='e({},{})'.format(l,a))
         for l in V for a in N}
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
    # 13)
    for j in F:
        for l in V:
            m.addConstr(quicksum(z[l, i, j] for i in C) <= y[j], name='C_13_({},{})'.format(j, l))
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
    # 17)
    for i in C:
        for l in V:
            m.addConstr(quicksum(z[l, i_1, i] for i_1 in C + D if i != i_1) == quicksum(
                z[l, i, i_1] for i_1 in C if i != i_1) + quicksum(z[l, i, j] for j in F),
                        name='C_17_({},{})'.format(i, l))
    # 18)
    for i in C:
        for l in V:
            m.addConstr(quicksum(z[l, i, i_1] for i_1 in C if i != i_1) <= 1, name='C_18_({},{})'.format(i, l))
    # 19)
    for l in V:
        m.addConstr(quicksum(t[a, b] * z[l, a, b] for a in N for b in N if a != b) <= T[l], name='C_19_({})'.format(l))

    # 20) no loop over same node constraint:
    for l in V:
        for i in N:
            m.addConstr(z[l, i, i] == 0, name='C_20_no_loop({},{})'.format(l, i))

    # New constraints by Pizzari and Pinto in 23/12/21 binding z and h to behave
    # 21)
    for l in V:
        for a in C:
            m.addConstr(quicksum(z[l,a,b] for b in C+F) == h[l,a], name='C_21_fromclient_to_clientORfacility_({},{})'.format(l,a))
    # 22)
    for l in V:
        for a in C:
            m.addConstr(quicksum(z[l,b,a] for b in D+C) == h[l,a], name='C_22_toclient_from_clientORdeposit_({},{})'.format(l,a))

    for l in V:
        for a in D+C:
            for b in C+F:
                m.addConstr((z[l,a,b]==1)>>(e[l,b]==e[l,a]+1), name='ordine({},{})'.format(l,a,b))

    for l in V:
        for a in D:
            m.addConstr(e[l,a]==0, name='ordine_dep({},{})'.format(l,a))

    for l in V:
        for a in N:
            m.addConstr(e[l,a]<=M*h[l,a], name='controllo_ordine({},{})'.format(l,a))
    # frase per commit

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
        for var in m.getVars():
            vars_opt.append([var.VarName, var.x])
            if var.VarName.startswith('h'):
                h_opt_dict[eval(var.VarName[2:-1])] = var.x
            if var.VarName.startswith('z'):
                z_opt_dict[eval(var.VarName[2:-1])] = var.x
            if var.VarName.startswith('e'):
                e_opt_dict[eval(var.VarName[2:-1])] = var.x

        vars_opt = pd.DataFrame.from_records(vars_opt, columns=["variable", "value"])
        vars_opt.to_excel('risultati_SP.xlsx')

        h_opt = vars_opt[vars_opt['variable'].str.contains("h", na=False)]
        z_opt = vars_opt[vars_opt['variable'].str.contains("z", na=False)]
        e_opt = vars_opt[vars_opt['variable'].str.contains("e", na=False)]

        h_opt['value'].apply(pd.to_numeric).astype(int)
        z_opt['value'].apply(pd.to_numeric).astype(int)
        e_opt['value'].apply(pd.to_numeric).astype(int)

        df_vars_list = [h_opt, z_opt, e_opt]

        opt_vars = {'h': h_opt_dict, 'z': z_opt_dict, 'e': e_opt_dict}

        return opt_vars, df_vars_list


# scorro le facility, inizialmente con la taglia + piccola e le apro fin quando ci è budget,
# se avanza budjet ampio le taglie fin quando ci è buget con un nuova passata su quelle aperte,
# se avanza bujet ampio alla ultima taglia possibile sempre su quella aperte
def get_feasible_sol_SP(F, H, FCost, B):
    y = {}
    y_size = {}
    y_totcost = 0
    h = 'S'
    for j in F:
        if y_totcost + FCost[h] <= B:
            y[j] = 1
            y_size[j] = h
            y_totcost += FCost[h]
        else:
            y[j] = 0
    h = 'M'
    for j in F:
        if y[j] == 1 and y_totcost + FCost[h] - FCost['S'] <= B:
            y_size[j] = h
            y_totcost += FCost[h]
            y_totcost -= FCost['S']
    h = 'L'
    for j in F:
        if y[j] == 1 and y_totcost + FCost[h] - FCost['M'] <= B:
            y_size[j] = h
            y_totcost += FCost[h]
            y_totcost -= FCost['S']

    return y, y_size, y_totcost

def get_facility_load(OP_opt_vars, SP_opt_vars_init):

    h = OP_opt_vars['h']
    y = SP_opt_vars_init['y']

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


if __name__ == '__main__':
    # y_0, y_0_size, y_0_totcost = get_feasible_sol_SP(F,H,FCost,B)
    first_try = False

    time_limit = 30
    gap_tol = 0.05  # 1e-5

    OP_opt_vars = None

    print('########################### \n FIRST ATTEMPT TO SOLVE SP \n########################### ')
    get_single_sol_SP = True
    SP_opt_vars_init, _, _ = SP_model(params, gamma, OP_opt_vars, gap_tol, time_limit,
                                                                    get_single_sol_SP)
    get_single_sol_SP = False

    print('########################### \n FIRST ATTEMPT TO SOLVE OP \n###########################')
    OP_opt_vars, OP_vars_list = OP_model(params, SP_opt_vars_init, gap_tol, time_limit, first_try)

    # evaluate load of open facility due to trucks routing
    j_load = get_facility_load(OP_opt_vars, SP_opt_vars_init)
    # update and fix var. n according to actual j_load

    r = SP_opt_vars_init['r']
    open_list = [r_var for r_var, value in r.items() if value == 1]

    x = SP_opt_vars_init['x']
    d_cluster = {}
    for s in S:
        demand_s = 0
        for i in C:
            demand_s += x[i,s]*d[i]
        d_cluster[s] = demand_s

    capf_usage = {}
    for r_jhs in open_list:
        j = r_jhs[0]
        h = r_jhs[1]
        free_capf = capf[(j,h)] - j_load[j]
        if free_capf >= sc[(j,h)]:
            SP_opt_vars_init['n'][j] = 0
        elif free_capf > 0 and free_capf < sc[(j,h)]:
            SP_opt_vars_init['n'][j] = (sc[(j,h)] - free_capf)/sc[(j,h)]
        else: # free_capf < 0 --> infeasible!
            break


    print('########################### \n Evaluation of total SP objective value \n###########################')
    _, SP_optval, SP_vars_list = SP_model(params, gamma, OP_opt_vars, gap_tol, time_limit, get_single_sol_SP,
                                         SP_opt_vars_init)





    Heuristic = False
    if Heuristic:
        # Heuristic start:
        x = SP_opt_vars['x']
        y = SP_opt_vars['y']
        n = SP_opt_vars['n']
        r = SP_opt_vars['r']

        B_used = sum(c[j, h] * r[j, h, s] for j in F for h in H for s in S)

        dist = {(a, b): disdur[(a, b)]['duration'] for (a, b) in itertools.product(N, N)}


        def get_closest_facility(j_0):
            distances = {}
            for j in F:
                if j != j_0:
                    distances[j] = dist[(j_0, j)]
            return max(distances, key=distances.get)


        open_list = [r_var for r_var, value in r.items() if value == 1]  # list of all (j,h,s) s.t. r[j,h,s] = 1

        ###################################################################
        ## 1 step: enlarge size of open facilities using safety stock ##
        if max(n.values()) > 0:  # at least one facility is using safety stock --> leader cost to decrease
            ss_used_list = [(n_var, value) for n_var, value in n.items() if
                            value > 0]  # list of all facilities j with n_j > 0
            ss_used_list.sort(key=lambda x: x[1], reverse=True)  # list is sort in descending oreder w.r.t n_j
            for elem in ss_used_list:  # for each of these facilities where safaty stock is used (in descending oreder w.r.t n_j)
                j = elem[0]
                for s in S:
                    for h in H:
                        if (j, h, s) in open_list and h != 3:  # upgrade its size if possible
                            if B_used + c[j, h + 1] - c[j, h] <= B:  # check if budjet is available
                                r[j, h, s] = 0
                                r[j, h + 1, s] = 1
                                B_used += c[j, h + 1] - c[j, h]
                                print('facility j = ' + str(j) + ' has been enlarged to size ' + str(h + 1))

                        ############################################################
                        ## 2 step: attempt to modify Y by closing or opening some ##
                        ## 2.1 step: computer capacity utilization of each open facility according to follower behaviour Z
                        # for elem in ss_used_list:

                        else:  # open the facility closest to j
                            j_to_open = get_closest_facility(j)
                            y[j_to_open] = 1
                            r[j_to_open, 1, s] = 1  # at its minimum size

                            # --> passare la nuova y a OP ?


        else:  # cluster definition may be inefficient --> attempt to change facility distribution across clusters
            print('ciao')