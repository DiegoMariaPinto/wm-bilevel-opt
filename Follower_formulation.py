import itertools
import random

from gurobipy import *
from gurobipy import quicksum
from gurobipy import GRB
from gurobipy import Model

import pandas as pd
import numpy as np
from Get_instances import load_json_instance


def OP_model(params, SP_vars, gap_tol, time_limit):
    m = Model('SP')

    # PARAMS

    gamma = params['generic_params']['gamma']
    d = params['generic_params']['d']
    OP_params = params['OP_params']

    F = params['generic_params']['F']
    C = params['generic_params']['C']
    D = params['generic_params']['D']
    N = params['generic_params']['N']
    V = params['generic_params']['V']
    NC = params['generic_params']['NC']

    t = OP_params['t']
    truck_em_coeff = OP_params['truck_em_coeff']
    em_t = OP_params['em_t']
    cv = OP_params['cv']
    T = OP_params['T']
    P = OP_params['P']
    a_matrix = OP_params['a_matrix']

    SP_params = params['SP_params']
    H = SP_params['H']
    capf = SP_params['capf']
    S = SP_params['S']

    y = SP_vars['y']
    r = SP_vars['r']

    big_M = max(cv.values())

    # OP Variables

    h = {(l, a): m.addVar(vtype=GRB.BINARY, name='h({},{})'.format(l, a))
         for l in V for a in N}

    z = {(l, a, b): m.addVar(vtype=GRB.BINARY, name='z({},{},{})'.format(l, a, b))
         for l in V for a in N for b in N}

    e = {(l,a): m.addVar(lb=0, vtype=GRB.INTEGER, name='e({},{})'.format(l,a))
         for l in V for a in N}

    v = {(l,j): m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='v({},{})'.format(l,j))
         for l in V for j in F}

    rho = {(l,a): m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='rho({},{})'.format(l,a))
    for l in V for a in C}


    # obj.fun. linearization var
    w = m.addVar(vtype=GRB.CONTINUOUS, obj=1, name='w')

    # Constraints
    # (10)
    for l in V:
        m.addConstr(quicksum(h[l, j] for j in F) == quicksum(z[l, k, i] for k in D for i in C),
                     name='C_10_({})'.format(l))
    # (11)
    for l in V:
        m.addConstr(quicksum(h[l, k] for k in D) == 1, name='C_11_onedeposit({})'.format(l))
    # (12)
    for l in V:
        m.addConstr(quicksum(h[l, j] for j in F) == 1, name='C_12_onefacility({})'.format(l))
    # (13)
    for i in C:
        m.addConstr(d[i] == quicksum(rho[l,i] for l in V), name = 'C_13_demand_satisfied({})'.format(i))
    # (14)
    for l in V:
        for i in C:
            m.addConstr(rho[l,i] <= big_M*h[l,i], name='C_14_pickup_iff_visited({},{})'.format(l,i))
    # (15)
    for l in V:
        for i in C:
            m.addConstr(rho[l, i] >= 0.1*d[i]*h[l, i], name='C_15_minimum_pickup({},{})'.format(l, i))
    # (16)
    for l in V:
        m.addConstr(quicksum(rho[l,i] for i in C) <= cv[l], name='C_16_capacity_vehicle({})'.format(l))
    # (17)
    for l in V:
        m.addConstr(quicksum(t[a, b] * z[l, a, b] for a in N for b in N if a != b) <= T[l],
                    name='C_17_time_vehicle({})'.format(l))
    # (18)
    for k in D:
        for l in V:
            m.addConstr(quicksum(z[l, k, i] for i in C) <= a_matrix[k, l], name='C_18_exit_from_depot({},{})'.format(k, l))
    # (19)
    for k in D:
        for l in V:
            m.addConstr(quicksum(z[l, j, k] for j in F) <= a_matrix[k, l], name='C_19_enter_in_depot({},{})'.format(k, l))
    # (20)
    for j in F:
        for l in V:
            m.addConstr(quicksum(z[l, j, i] for i in C) == 0, name='C_20_noclient_after_facility({},{})'.format(j, l))
    # (21)
    for j in F:
        for l in V:
            m.addConstr(quicksum(z[l, j, k] for k in D) == quicksum(z[l, i, j] for i in C),
                        name='C_21_depot_after_facility({},{})'.format(j, l))
    # (22)
    for l in V:
        for a in C:
            m.addConstr(quicksum(z[l,a,b] for b in C+F) == h[l,a], name='C_22_fromclient_to_clientORfacility_({},{})'.format(l,a))
    # (23)
    for l in V:
        for a in C:
            m.addConstr(quicksum(z[l,b,a] for b in D+C) == h[l,a], name='C_23_toclient_from_clientORdeposit_({},{})'.format(l,a))
    ### no loop constraints:
    # (24)
    for l in V:
        for a in D + C:
            for b in C + F:
                m.addConstr((z[l, a, b] == 1) >> (e[l, b] == e[l, a] + 1), name='C_24_ordine({},{})'.format(l, a, b))
    # (25)
    for l in V:
        for a in D:
            m.addConstr(e[l, a] == 0, name='C_25_ordine_dep({},{})'.format(l, a))
    # (26)
    for l in V:
        for a in N:
            m.addConstr(e[l, a] <= NC * h[l, a], name='C_26_controllo_ordine({},{})'.format(l, a))
    # (27)
    for l in V:
        for j in F:
            m.addConstr((h[l,j]==1)>>(quicksum(rho[l,i] for i in C) == v[l,j]), name='C_27_load_of_l_to_j({},{})'.format(l,j))
    # (28)
    for j in F:
            m.addConstr(quicksum(v[l,j] for l in V) <= quicksum(r[j,h,s]*capf[j,h] for h in H for s in S), name='load_of_j({})'.format(j))
    # (29)
    for l in V:
        for j in F:
            m.addConstr((h[l,j]==0)>>(v[l,j]==0), name='C_29_no_load_unvisited_facility({},{})'.format(l,j))
    # NEW linearization related constraints
    for k in D:
        m.addConstr(
            w >= quicksum(z[l, a, b] * t[a, b] for a in N for b in N for l in V if a_matrix[k, l] == 1)
            , name='C_27_({})'.format(k))
    # (9)
    # for i in C:
    #   m.addConstr(quicksum(h[l, i] for l in V) >= 1, name='C_9_({})'.format(i))
    # # 13)
    # for j in F:
    #     for l in V:
    #         m.addConstr(quicksum(z[l, i, j] for i in C) <= y[j], name='C_13_({},{})'.format(j, l))


    # 15)
    #for j in F:
    #    for l in V:
    #        m.addConstr(quicksum(z[l, i, j] for i in C) <= h[l, j], name='C_15_({},{})'.format(j, l))

    # # 17)
    # for i in C:
    #     for l in V:
    #         m.addConstr(quicksum(z[l, i_1, i] for i_1 in C + D if i != i_1) == quicksum(
    #             z[l, i, i_1] for i_1 in C if i != i_1) + quicksum(z[l, i, j] for j in F),
    #                     name='C_17_({},{})'.format(i, l))
    # 18)
    #for i in C:
    #    for l in V:
    #        m.addConstr(quicksum(z[l, i, i_1] for i_1 in C if i != i_1) <= 1, name='C_18_({},{})'.format(i, l))


    # 20) no loop over same node constraint:
    # for l in V:
    #     for i in N:
    #         m.addConstr(z[l, i, i] == 0, name='C_20_no_loop({},{})'.format(l, i))


    ############################################################################################

    ############################################################################################
    #for i in C:
    #    m.addConstr(d[i] == quicksum(p[l,i] for l in V), 'soddisfa_domanda({})'.format(i))




    #for l in V:
    #    for i in C:
    #        m.addConstr(rho[l,i] >= p[l,i]-big_M*(1-h[l,i]), name = 'linearizzazione_p_2({},{})'.format(l, i))

    #for l in V:
    #    for i in C:
    #        m.addConstr(rho[l,i] <= p[l,i], name = 'linearizzazione_p_3({},{})'.format(l, i))



    #for l in V:
    #    for j in F:
    #        m.addConstr((h[l, j] == 0) >> (v[l, j] == 0),  name='load_of_l_not_to_j({},{})'.format(l, j))

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



    ################# solve the formulation ####################

    m.Params.MIPGap = gap_tol
    m.Params.TimeLimit = time_limit

    m.setParam('OutputFlag', 0)
    m.setParam('LogToConsole', 0)

    m.setParam(GRB.Param.OutputFlag, 0)
    m.setParam(GRB.Param.LogToConsole, 0)

    ###########################################
    m.setParam('OutputFlag', 1)
    m.setParam('LogToConsole', 1)

    m.setParam(GRB.Param.OutputFlag, 1)
    m.setParam(GRB.Param.LogToConsole, 1)
    ###########################################

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
        print('OP MODEL IS INFEASIBLE')
        m.computeIIS()
        for c in m.getConstrs():
            if c.IISConstr:
                print(c.constrName)
        m.write('OP.ilp')
        return 'infeasible', 'infeasible'

    if status == GRB.Status.OPTIMAL or status == 9:  # 9 is equivalent to 'Time Limit Reached'
        if status == 2:
            print('OP model was solved to optimality')
        if status == 9:
            print('OP Time Limit Reached')
        optObjVal = m.getAttr(GRB.Attr.ObjVal)
        bestObjBound = m.getAttr(GRB.Attr.ObjBound)
        Runtime = m.Runtime
        # Gap = m.MIPGap

        vars_opt = []
        h_opt_dict = {}
        z_opt_dict = {}
        e_opt_dict = {}
        v_opt_dict = {}
        p_opt_dict = {}
        rho_opt_dict = {}

        for var in m.getVars():
            vars_opt.append([var.VarName, var.x])
            if var.VarName.startswith('h'):
                h_opt_dict[eval(var.VarName[2:-1])] = round(var.x)
            elif var.VarName.startswith('z'):
                z_opt_dict[eval(var.VarName[2:-1])] = round(var.x)
            elif var.VarName.startswith('e'):
                e_opt_dict[eval(var.VarName[2:-1])] = round(var.x)
            elif var.VarName.startswith('v'):
                v_opt_dict[eval(var.VarName[2:-1])] = round(var.x)
            elif var.VarName.startswith('p'):
                v_opt_dict[eval(var.VarName[2:-1])] = round(var.x)
            elif var.VarName.startswith('rho'):
                v_opt_dict[eval(var.VarName[4:-1])] = round(var.x)

        vars_opt = pd.DataFrame.from_records(vars_opt, columns=["variable", "value"])
        vars_opt.to_excel('risultati_SP.xlsx')

        h_opt = vars_opt[vars_opt['variable'].str.startswith("h", na=False)]
        z_opt = vars_opt[vars_opt['variable'].str.contains("z", na=False)]
        e_opt = vars_opt[vars_opt['variable'].str.contains("e", na=False)]
        v_opt = vars_opt[vars_opt['variable'].str.contains("v", na=False)]
        p_opt = vars_opt[vars_opt['variable'].str.contains("p", na=False)]
        rho_opt = vars_opt[vars_opt['variable'].str.contains("rho", na=False)]

        h_opt['value'].apply(pd.to_numeric)
        z_opt['value'].apply(pd.to_numeric)
        e_opt['value'].apply(pd.to_numeric)
        v_opt['value'].apply(pd.to_numeric)
        p_opt['value'].apply(pd.to_numeric)
        rho_opt['value'].apply(pd.to_numeric)

        df_vars_list = [h_opt, z_opt, e_opt, v_opt, p_opt, rho_opt]

        opt_vars = {'h': h_opt_dict, 'z': z_opt_dict, 'e': e_opt_dict, 'v': v_opt_dict, 'p': p_opt_dict, 'rho': rho_opt_dict}

        return opt_vars, df_vars_list