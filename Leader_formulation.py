import itertools
import random

from gurobipy import *
from gurobipy import quicksum
from gurobipy import GRB
from gurobipy import Model

import pandas as pd
import numpy as np
from Get_instances import load_json_instance


def SP_model(params, OP_vars, gap_tol, time_limit, get_first_sol_SP=False, SP_vars_to_fix=None):

    m = Model('SP')

    # PARAMS

    F = params['generic_params']['F']
    C = params['generic_params']['C']
    D = params['generic_params']['D']
    N = params['generic_params']['N']
    V = params['generic_params']['V']

    gamma = params['generic_params']['gamma']
    d = params['generic_params']['d']
    SP_params = params['SP_params']

    H = SP_params['H']
    CAPH = SP_params['CAPH']
    EM = SP_params['EM']
    FCost = SP_params['FCost']
    capf = SP_params['capf']
    c = SP_params['c']
    B = SP_params['B']
    em_f = SP_params['em_f']
    NS = SP_params['NS']
    S = SP_params['S']
    sc = SP_params['sc']

    OP_params = params['OP_params']
    P = OP_params['P']
    em_t = OP_params['em_t']

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
    m.addConstr(quicksum(c[j, h] * r[j, h, s] for j in F for h in H for s in S) <= B, name='C6_budjet')

    if get_first_sol_SP:
        # (7)
        max_j_for_s = 2
        for s in S:
            m.addConstr(quicksum(r[j,h,s] for j in F for h in H) <= max_j_for_s, name='C7_min_clusters')


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

    m.setParam('OutputFlag', 0)
    m.setParam('LogToConsole', 0)

    m.setParam(GRB.Param.OutputFlag, 0)
    m.setParam(GRB.Param.LogToConsole, 0)

    m.Params.MIPGap = gap_tol
    m.Params.TimeLimit = time_limit

    m.modelSense = GRB.MINIMIZE
    m.update()
    m.write('SP_model.lp')



    m.optimize()

    status = m.status
    if status == GRB.Status.INFEASIBLE:
        status = "infeasible"
        print('SP MODEL IS INFEASIBLE')
        m.computeIIS()
        for c in m.getConstrs():
            if c.IISConstr:
                print(c.constrName)
        m.write('SP.ilp')
        return np.nan, np.nan, np.nan

    if status == GRB.Status.OPTIMAL or status == 9:  # 9 is equivalent to 'Time Limit Reached'
        if status == 2:
            print('SP model was solved to optimality')
        if status == 9:
            print('SP Time Limit Reached')
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
            elif var.VarName.startswith('r'):
                r_opt_dict[eval(var.VarName[2:-1])] = round(var.x)
            elif var.VarName.startswith('y'):
                y_opt_dict[eval(var.VarName[2:-1])] = round(var.x)
            elif var.VarName.startswith('n'):
                n_opt_dict[eval(var.VarName[2:-1])] = var.x
            elif var.VarName.startswith('q'):
                q_opt_dict[eval(var.VarName[2:-1])] = round(var.x)

        vars_opt = pd.DataFrame.from_records(vars_opt, columns=["variable", "value"])

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