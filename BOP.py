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
H = ['Small', 'Medium', 'Large']
CAPH = {'Small':500, 'Medium':1000, 'Large':1500}
EM = {'Small':20, 'Medium':25, 'Large':30} # test different shapes
FCost = {'Small':100, 'Medium':150, 'Large':180}
capf  = {(j,h): CAPH[h] for (j, h) in itertools.product(F,H)}
c = {(j,h): FCost[h] for (j, h) in itertools.product(F,H)}
B = 3000 # budjet for facilities opening  # 1600
em_f = {(j,h):EM[h] for (j,h) in itertools.product(F,H)}
NS = NF # upper bound of the number of clusters
S = list(range(NS))
# sc is the safe capacity for each facility j of size h
sc = {(j,h): 0.2*CAPH[h] for (j,h) in itertools.product(F,H)}

#SP_params = {'H':H, 'CAPH' = CAPH, etc..}
#########################
# OP specific parameters
t  = {(a,b):disdur[(a,b)]['duration'] for (a,b) in itertools.product(N,N)}
truck_em_coeff = 1.2
em_t = {(a,b): truck_em_coeff*disdur[(a,b)]['distance'] for (a,b) in itertools.product(N,N)}
random_cv = np.random.randint(30,60,NV).tolist() # vehicles capacities
cv = {l: random_cv[l] for l in V}
random_T = np.random.randint(6,9,NV).tolist() # maximum servicing times per tour (electic or combustion) 3,8
T = {l: random_T[l] for l in V}
random_P = np.random.randint(50,100,NF).tolist() # maximum penalty for a facility
P = {j: random_P[j] for j in F}
a_matrix = {(k,l) : 1 for (k,l) in itertools.product(D,V) if k != l} # trucks distribution across the depots
# a_k_l == 1 if truck l start its tour from depot k

# OP_params = {....}
#########################

# demand vector d= ?
random_d = np.random.randint(100,250,NC).tolist() # demand of clients i in C
d = {i: random_d[i-NF] for i in C}

# gamma vector of gamma_1,2,3
params = []
gamma = {'1':0.3,'2':0.3,'3':0.4}

def SP_model(params, gamma, OP_vars, gap_tol, time_limit, get_single_sol_SP = False):

    m = Model('SP')

    if get_single_sol_SP:
        z = None
    else:
        z = OP_vars['z']

    # Variables

    x = {(i, s): m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x({},{})'.format(i, s))
         for i in C for s in S}

    r = {(j, h, s): m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='r({},{},{})'.format(j, h, s))
         for j in F for h in H for s in S}

    y = {(j): m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y({})'.format(j))
         for j in F}

    n = {(j): m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='n({})'.format(j))
         for j in F}

    # obj.fun. linearization vars
    if not get_single_sol_SP:
        g = {(a,b,l,s): m.addVar(lb=0, vtype=GRB.CONTINUOUS, name ='g({},{},{},{})'.format(a,b,l,s))
             for a in C for b in C for l in V for s in S}

        u = {(a,b,s): m.addVar(lb=0,ub=1, vtype=GRB.BINARY, name = 'u({},{},{})'.format(a,b,s))
             for a in C for b in C for s in S}


    # Constraints
    # (1)
    for i in C:
        m.addConstr(quicksum(x[i,s] for s in S) == 1, name='C1_client_to_cluster({})'.format(i))
    # (2)
    for j in F:
        m.addConstr(quicksum(r[j,h,s] for s in S for h in H) == y[j], name='C2_one_size_only({})'.format(j))
    # (3)
    for s in S:
        m.addConstr(quicksum(x[i,s]*d[i] for i in C) <= quicksum(r[j,h,s]*capf[j,h] - (r[j,h,s]-n[j])*sc[j,h] for j in F for h in H), name='C3_demand_to_cluster({})'.format(s))
    # (4)
    for i in C:
        for s in S:
            m.addConstr(x[i,s] <= quicksum(r[j,h,s] for j in F for h in H), name='C4_facility_to_cluster({},{})'.format(i, s))
    # (5)
    # for s in S:
    #     m.addConstr(quicksum(x[i, s] for i in C) <= quicksum(r[j, h, s] for j in F for h in H), name='C5_empty_cluster({})'.format(s))

    c = {(j, h): FCost[h] for (j, h) in itertools.product(F, H)}
    # (6)
    m.addConstr(quicksum(c[j,h]*r[j,h,s] for j in F for h in H for s in S) <= B, name='C6_budjet_constr')

    # linearization related constraints

    if not get_single_sol_SP:
        # (21): g var lb is included in var definition
        # (22)
        for a in C:
            for b in C:
                for l in V:
                    for s in S:
                        m.addConstr(g[a,b,l,s] >= z[l,a,b] - u[a,b,s])
        # (23)
        for a in C:
            for b in C:
                for s in S:
                    m.addConstr(u[a,b,s] >= x[a,s] + x[b,s] - 1)
        # (24)
        for a in C:
            for b in C:
                for s in S:
                    m.addConstr(u[a,b,s] <= (x[a,s] + x[b,s])/2)

    # objective function

    if get_single_sol_SP:
        # risolvo con un pezzo di fo ed i vincoli da 1 a 6, prendo le y e risolvo il follower.
        # ora provo a valorizzare la fo del leader
        # cambiando eursitcamente le y e valutando il risultato in termini di fo
        m.setObjective(gamma['1'] * quicksum(r[j, h, s] * em_f[j, h] for j in F for h in H for s in S) +
                       gamma['2'] * quicksum(n[j] * P[j] for j in F))
    else:
        m.setObjective(gamma['1']*quicksum(z[l,a,b]*em_t[a,b] for a in N for b in N for l in V) +
                       gamma['1']*quicksum(r[j,h,s]*em_f[j,h] for j in F for h in H for s in S) +
                       gamma['2']*quicksum(n[j]*P[j] for j in F) +
                       gamma['3']*quicksum(g[a,b,l,s] for a in C for b in C for l in V for s in S))



    ################# solve the formulation ####################

    m.Params.MIPGap = gap_tol
    m.Params.TimeLimit = time_limit

    m.modelSense = GRB.MINIMIZE
    m.update()
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

    if status == GRB.Status.OPTIMAL or status == 9: # 9 is equivalent to 'Time Limit Reached'
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
                r_opt_dict[var.VarName[2:-1]] = var.x
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

        df_vars_list = [x_opt,r_opt,y_opt,n_opt]

        opt_vars = {'x': x_opt_dict, 'r': r_opt_dict, 'y': y_opt_dict, 'n': n_opt_dict}
        
        return opt_vars, optObjVal, df_vars_list


def OP_model(params, SP_vars, gap_tol, time_limit, first_try, y_0 = None):
    m = Model('SP')

    if first_try == True:
        y = y_0
    else:
        y = SP_vars['y']

    # OP Variables

    h = {(l, a): m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='h({},{})'.format(l, a))
         for l in V for a in N}

    z = {(l,a,b): m.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z({},{},{})'.format(l, a, b))
         for l in V for a in N for b in N}

    # obj.fun. linearization var

    w = m.addVar(vtype=GRB.CONTINUOUS, obj=1, name = 'w')

    # Constraints
    # (7)
    for k in D:
        for l in V:
            m.addConstr(quicksum(h[l, j] for j in F) == quicksum(z[l,k,i] for k in D for i in C), name='C_7_({},{})'.format(k,l))
    # (8)
    for i in C:
        m.addConstr(quicksum(h[l,i] for l in V) == 1, name='C_8_({})'.format(i))
    # (9)
    for l in V:
        m.addConstr(quicksum(h[l,k] for k in D) == 1, name='C_9_({})'.format(l))
    # (10)
    for l in V:
        m.addConstr(quicksum(h[l,i]*d[i] for i in C) <= cv[l], name='C_10_({})'.format(l))
    # 11)
    for k in D:
        for l in V:
            m.addConstr(quicksum(z[l,k,i] for i in C) <= a_matrix[k,l], name='C_11_({},{})'.format(k,l))
    # 12)
    for j in F:
        for l in V:
            m.addConstr(quicksum(z[l,i,j] for i in C) <= y[j] , name='C_12_({},{})'.format(j,l))
    # 13)
    for j in F:
        for l in V:
            m.addConstr(quicksum(z[l,j,i] for i in C) == 0, name='C_13_({},{})'.format(j,l))
    # 14)
    for j in F:
        for l in V:
            m.addConstr(quicksum(z[l,i,j] for i in C) <= h[l,j], name='C_14_({},{})'.format(j,l))
    # 15)
    for j in F:
        for l in V:
            m.addConstr(quicksum(z[l,j,k] for k in D) == quicksum(z[l,i,j] for i in C), name='C_15_({},{})'.format(j,l))
    # 16)
    for i in C:
        for l in V:
            m.addConstr(quicksum(z[l,i_1,i] for i_1 in C+D if i != i_1) == quicksum(z[l,i,i_1] for i_1 in C if i != i_1) + quicksum(z[l,i,j] for j in F), name='C_16_({},{})'.format(i,l))
    # 17)
    for i in C:
        for l in V:
            m.addConstr(quicksum(z[l,i,i_1] for i_1 in C if i != i_1) <= 1, name='C_17_({},{})'.format(i,l))
    # 18)
    for l in V:
        m.addConstr(quicksum(t[a,b]*z[l,a,b] for a in N for b in N if a != b) <= T[l], name='C_18_({})'.format(l))

    # linearization related constraints
    # (27)
    for k in D:
        for k_1 in D:
            if k_1 != k:
                m.addConstr( w >= quicksum(z[l,a,b]*t[a,b] for a in N for b in N for l in V if a_matrix[k,l] == 1) -
                                  quicksum(z[l,a,b]*t[a,b] for a in N for b in N for l in V if a_matrix[k_1,l] == 1)
                                    , name='C_27_({},{})'.format(k,k_1))

    ################# solve the formulation ####################

    m.Params.MIPGap = gap_tol
    m.Params.TimeLimit = time_limit

    m.modelSense = GRB.MINIMIZE
    m.update()

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
        for var in m.getVars():
            vars_opt.append([var.VarName, var.x])
            if var.VarName.startswith('h'):
                h_opt_dict[eval(var.VarName[2:-1])] = var.x
            if var.VarName.startswith('z'):
                z_opt_dict[eval(var.VarName[2:-1])] = var.x

        vars_opt = pd.DataFrame.from_records(vars_opt, columns=["variable", "value"])

        opt_vars = {'h': h_opt_dict, 'z': z_opt_dict}

        return opt_vars

# scorro le facility, inizialmente con la taglia + piccola e le apro fin quando ci è budget,
# se avanza budjet ampio le taglie fin quando ci è buget con un nuova passata su quelle aperte,
# se avanza bujet ampio alla ultima taglia possibile sempre su quella aperte
def get_feasible_sol_SP(F,H,FCost,B):

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


if __name__ == '__main__':
    # y_0, y_0_size, y_0_totcost = get_feasible_sol_SP(F,H,FCost,B)
    first_try = False

    time_limit = 60
    gap_tol = 0.05 # 1e-2

    OP_opt_vars = None

    print('########################### \n FIRST ATTEMPT TO SOLVE SP \n########################### ')
    get_single_sol_SP = True
    SP_opt_vars_start, SP_optval_start, df_vars_list_start = SP_model(params, gamma, OP_opt_vars, gap_tol, time_limit, get_single_sol_SP )
    get_single_sol_SP = False

    print('########################### \n FIRST ATTEMPT TO SOLVE OP \n########################### ')
    OP_opt_vars = OP_model(params, SP_opt_vars_start, gap_tol, time_limit, first_try)

    print('########################### \n SECOND ATTEMPT TO SOLVE SP \n########################### ')
    SP_opt_vars_new, SP_optval_new, df_vars_list_new = SP_model(params, gamma, OP_opt_vars, gap_tol, time_limit, get_single_sol_SP)














