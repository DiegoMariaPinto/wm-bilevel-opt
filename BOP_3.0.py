import itertools
import random

from gurobipy import *
from gurobipy import quicksum
from gurobipy import GRB
from gurobipy import Model

import pandas as pd
import numpy as np
from Get_instances import load_json_instance
from Leader_formulation import SP_model
from Follower_formulation import OP_model


def create_params(NF,NC,ND,NV,disdur,test_realistic_inst):

    np.random.seed(0)
    random.seed((0))

    # corresponding sets:
    F = list(range(NF))  # facility list
    C = list(range(NF, NC + NF))
    D = list(range(NC + NF, ND + NC + NF))
    N = F + C + D
    V = list(range(NV))

    generic_params = {'F': F, 'C': C, 'D': D, 'N': N, 'V': V, 'NF': NF, 'NC':NC, 'ND': ND, 'NV': NV}

    ########################
    # SP specific parameters

    if test_realistic_inst:
        realistic_capacity = pd.read_excel('BOP_realistic_instance_v2.xlsx', sheet_name='facility').astype({"fraction_cap_daily M": int}, errors='raise')['fraction_cap_daily M'].to_list()
        capacity_dict = { i : realistic_capacity[i] for i in range(0, len(realistic_capacity)) }
        H = [1, 2, 3]  # 1 = S, 2 = M, 3 = L
        CAPH = {1: 0.7, 2: 1, 3: 1.3}  # 1 reduces cap of 30%, 2 set its default cap, 3 enlarge cap of 30%
        capf = {(j, h): int(capacity_dict[j]*CAPH[h]) for (j,h) in itertools.product(F, H)}
        # sc is the safe capacity for each facility j of size h
        sc = {(j, h): int(0.3 * capf[(j,h)]) for (j, h) in itertools.product(F, H)}

    else:
        H = [1, 2, 3]  # 1 = S, 2 = M, 3 = L
        CAPH = {1: 500, 2: 1000, 3: 1500}
        capf = {(j, h): CAPH[h] for (j, h) in itertools.product(F, H)}
        # sc is the safe capacity for each facility j of size h
        sc = {(j, h): 0.3 * CAPH[h] for (j, h) in itertools.product(F, H)}

    EM = {1: 20, 2: 25, 3: 30}  # test different shapes
    FCost = {1: 100, 2: 150, 3: 180}  # facility installation cost
    c = {(j, h): FCost[h] for (j, h) in itertools.product(F, H)}
    B = 3000  # budjet for facilities opening  # 1600
    em_f = {(j, h): EM[h] for (j, h) in itertools.product(F, H)}
    NS = NF  # upper bound of the number of clusters
    S = list(range(NS))


    SP_params = {'H':H, 'CAPH': CAPH, 'EM': EM, 'FCost': FCost, 'capf': capf, 'c':c, 'B':B, 'em_f': em_f, 'NS':NS, 'S': S, 'sc': sc}

    #########################
    # OP specific parameters

    if test_realistic_inst:
        # demand vector d
        realistic_demand = pd.read_excel('BOP_realistic_instance_v2.xlsx', sheet_name='client').astype({"demand_weekly": int}, errors='raise')['demand_weekly'].to_list()
        d = {NF + i: realistic_demand[i] for i in range(0, len(realistic_demand))}
        cv = {l: 40 for l in V} # 25 (##10) ton of capacity for each truck
        random_T = np.random.randint(5*60, 8*60, NV).tolist()  # maximum servicing times per tour (electic or combustion) 5,8 h * 60 minutes
        T = {l: random_T[l] for l in V}

    else:
        # demand vector d
        random_d = np.random.randint(50, 80, NC).tolist()  # demand of clients i in C
        d = {i: random_d[i - NF] for i in C}
        random_cv = np.random.randint(200, 400, NV).tolist()  # vehicles capacities
        cv = {l: random_cv[l] for l in V}
        random_T = np.random.randint(600, 900, NV).tolist()  # maximum servicing times per tour (electic or combustion) 3,8
        T = {l: random_T[l] for l in V}

    t = {(a, b): disdur[(a, b)]['duration'] for (a, b) in itertools.product(N, N)}
    truck_em_coeff = 1.2
    em_t = {(a, b): truck_em_coeff * disdur[(a, b)]['distance'] for (a, b) in itertools.product(N, N)}
    random_P = np.random.randint(50, 100, NF).tolist()  # maximum penalty for a facility
    P = {j: random_P[j] for j in F}
    M = 500
    u = {l: 90 + (l * 1) for l in V}

    ######################################################
    # distribute truck across depots s.t. a_k_l == 1 if truck l start its tour from depot k
    a_matrix = {(k, l): 1 for (k, l) in itertools.product(D, V) if
                k - D[0] == l - V[0]}  # trucks distribution across the depots

    for (k, l) in itertools.product(D, V):
        if k - D[0] != l - V[0]:
            a_matrix[(k, l)] = 0

    if len(V) > len(D):
        for i in np.arange(1,1+int(len(V)/len(D))):
            for (k, l) in itertools.product(D, V[i*len(D):]):
                if k - D[0] == l - V[i*len(D):][0]:
                    a_matrix[(k, l)] = 1

    ######################################################

    OP_params = {'t': t, 'truck_em_coeff': truck_em_coeff, 'em_t':em_t, 'cv': cv, 'T': T, 'P':P, 'a_matrix': a_matrix, 'M': M, 'u':u}
    #########################

    # gamma vector of gamma_1,2,3
    gamma = {'1': 0.3, '2': 0.3, '3': 0.4}

    generic_params['d'] = d
    generic_params['gamma'] = gamma

    params = {'SP_params': SP_params, 'OP_params': OP_params, 'generic_params': generic_params}

    return params


def get_facility_load(OP_opt_vars, SP_opt_vars, params):

    F = params['generic_params']['F']
    C = params['generic_params']['C']
    V = params['generic_params']['V']
    d = params['generic_params']['d']

    h = OP_opt_vars['h']
    y = SP_opt_vars['y']
    e = OP_opt_vars['e']

    loads = {}
    for l in V:
        load = 0
        for i in C:
            load += e[(l, i)]
        loads[l] = load

    j_load = {}
    for j in F:
        if y[j] == 1:
            load_j = 0
            for l in V:
                load_j += h[l, j] * loads[l]

            j_load[j] = load_j

    return j_load

def get_cluster_load(SP_opt_vars, params):

    C = params['generic_params']['C']
    d = params['generic_params']['d']
    S = params['SP_params']['S']

    x = SP_opt_vars['x']
    s_load = {}
    for s in S:
        demand_s = 0
        for i in C:
            demand_s += x[i, s] * d[i]
        s_load[s] = demand_s

    return s_load


def get_closest_facility_list(j_0, F, dist):
    distances = {}
    for j in F:
        if j != j_0:
            distances[j] = dist[(j_0, j)]

    closest_facility_list = [(j, dist_value) for j, dist_value in
                             distances.items()]  # list of all facilities j and their distance from j_0 input
    closest_facility_list.sort(key=lambda x: x[1], reverse=False)  # list is sort in descending order w.r.t n_j

    return closest_facility_list


def evalute_SP_objval(OP_opt_vars,SP_opt_vars, params, gap_tol):

    SP_params = params['SP_params']
    capf = SP_params['capf']
    sc = params['SP_params']['sc']

    # evaluate load of open facility due to trucks routing
    j_load = get_facility_load(OP_opt_vars, SP_opt_vars, params)
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


    _, SP_optval, __ = SP_model(params, OP_opt_vars, gap_tol, SP_time_limit, False, SP_opt_vars)

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


def heuristic(instance_name, maxit, SP_time_limit, OP_time_limit, test_realistic_inst):

    data = load_json_instance('./instances', instance_name + '.json')
    inst_data = data['inst_data']
    disdur = data['disdur_dict']

    gap_tol = 0.0025  # 1e-5

    NF = inst_data['NF']
    NC = inst_data['NC']
    ND = inst_data['ND']
    NV = inst_data['NV']


    params = create_params(NF, NC, ND, NV, disdur, test_realistic_inst)

    F = params['generic_params']['F']
    N = params['generic_params']['N']

    SP_params = params['SP_params']
    capf = SP_params['capf']
    sc = params['SP_params']['sc']

    H = SP_params['H']
    c = SP_params['c']
    B = SP_params['B']
    S = SP_params['S']

    OPs_OptGap_dict = {}
    ########################

    OP_opt_vars = None

    print('########################### \n FIRST ATTEMPT TO SOLVE SP \n########################### ')
    get_first_sol_SP = True
    SP_opt_vars_init, _, SP_vars_init_list = SP_model(params, OP_opt_vars, gap_tol, SP_time_limit,
                                                                    get_first_sol_SP)
    get_first_sol_SP = False

    # evaluate load of clusters due to clients assignement
    s_load = get_cluster_load(SP_opt_vars_init, params)
    print('########################### \n FIRST ATTEMPT TO SOLVE OP \n###########################')
    OP_opt_vars, OP_vars_list, OP_OptGap, OP_optval = OP_model(params, SP_opt_vars_init, gap_tol, OP_time_limit)

    OPs_OptGap_dict[0] = OP_OptGap
    OP_obj_evolution = {}
    OP_obj_evolution[0] = OP_optval

    # evaluate load of open facility due to trucks routing
    j_load = get_facility_load(OP_opt_vars, SP_opt_vars_init, params)

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
    SP_opt_vars, SP_optval, SP_vars_list = SP_model(params, OP_opt_vars, gap_tol, SP_time_limit, get_first_sol_SP,
                                         SP_opt_vars_init)

    first_eval = SP_optval

    SP_obj_evolution = {}
    SP_obj_evolution[0] = SP_optval
    best_obj = SP_optval


    print('first solution open facility list is : \n')
    print(open_list)

    dist = {(a, b): disdur[(a, b)]['duration'] for (a, b) in itertools.product(N, N)}

    Heuristic = True
    if Heuristic:
        print('####### Heuristic START HERE #######')
        trigger_0_tabu_list = []
        trigger_1_tabu_list = []
        trigger_2_tabu_list = []
        for count in range(1,maxit+1):

            print('Iteration n. '+str(count))
            print('########################### \n' + 'Heursitic Iteration n. ' + str(count) + '\n#########################')

            # Heuristic iteration start:
            x = SP_opt_vars['x']
            y = SP_opt_vars['y']
            r = SP_opt_vars['r']
            n = SP_opt_vars['n']

            n_before_attempt = n.copy()

            B_used = sum(c[j, h] * r[j, h, s] for j in F for h in H for s in S)
            open_list = [r_var for r_var, value in r.items() if value == 1]  # list of all (j,h,s) s.t. r[j,h,s] = 1

            ss_used_list = [(n_var, value) for n_var, value in n.items() if value > 0]  # list of all facilities j with n_j > 0
            ss_used_list.sort(key=lambda x: x[1], reverse=True)  # list is sort in descending order w.r.t n_j

            j_load = get_facility_load(OP_opt_vars, SP_opt_vars, params)
            j_usage = get_j_usage(open_list, j_load, capf)

            j_to_close_list = {k: v for (k, v) in j_usage.items() if v < 0.4}
            j_to_close_list = sorted(j_to_close_list.items(), key=operator.itemgetter(1), reverse=False)
            j_to_help_list = [keyval for keyval in ss_used_list if keyval[1] > 0.5]

            trigger = -1 # set trigger to none-value

            ################################################################
            ## 1 step: enlarge size of open facilities using safety stock ##
            if max(n.values()) > 0:  # at least one facility is using safety stock --> leader cost to decrease

                for elem in ss_used_list:  # for each of these facilities where safaty stock is used (in descending oreder w.r.t n_j)
                    j = elem[0][0]
                    h = elem[0][1]
                    s = elem[0][2]
                    if h != 3 and B_used + c[j, h + 1] - c[j, h] <= B:  # upgrade its size if possible and check if budjet is available
                        if (j,h,s) not in trigger_0_tabu_list:
                            r[j, h, s] = 0
                            r[j, h + 1, s] = 1
                            n[j, h, s] = 0
                            B_used += c[j, h + 1] - c[j, h]
                            open_list = [r_var for r_var, value in r.items() if value == 1]
                            print('Trigger 0: facility j = ' + str(j) + ' has been enlarged to size ' + str(h + 1))
                            trigger = 0
                            break

                #########################################################################################################
                # 2 step: attempt to modify Y by closing or opening some facility ##
                # This is done according to two differet triggers: 2.1 and 2.2

                # 2.1 step: compute percentage of stock usage w.r.t. facility capacity according to follower behaviour Z
                # close facility with lowest percentage use and open the closest facility to the one with biggest n value.

                if min(j_usage.values()) < 0.4 and trigger != 0:
                    trigger = 1
                    gotit = False

                    if not j_to_help_list: # if this list is actually empty stop
                        print('facility to help list is EMPTY -- heuristic stops here')
                        break

                    for j1 in j_to_close_list:
                        if gotit == True:
                            break
                        j_to_close = j1[0]
                        for j2 in j_to_help_list:
                            if gotit == True:
                                break
                            j_to_help = j2[0][0]
                            closest_facility_list = get_closest_facility_list(j_to_help, F, dist)
                            open_list = [r_var for r_var, value in r.items() if value == 1]
                            for j3 in closest_facility_list:
                                if j3[0] not in [elem[0] for elem in open_list]:
                                    if (j_to_close,j_to_help,j3[0]) not in trigger_1_tabu_list:
                                        j_to_open = j3[0]
                                        gotit = True
                                        break

                    if gotit == False:
                        print('All option explored -- heuristic stops here')
                        break

                    print('Trigger 1: found facility to close: is facility ' + str(j_to_close))
                    y[j_to_close] = 0
                    h_to_close, s_to_close = find_size_and_cluster_of_j(j_to_close, H, S, open_list)
                    r[j_to_close, h_to_close, s_to_close] = 0
                    print('Trigger 1: found facility to help: is facility ' + str(j_to_help))
                    y[j_to_open] = 1
                    r[j_to_open, h_to_close, s_to_close] = 1
                    print('Trigger 1: found facility to open: is facility ' + str(j_to_open))

                # 2.2 step: Help facility with the biggest ss usage by opening a new facility as close as possible
                elif max(n.values()) > 0.3 and trigger != 0:
                    trigger = 2
                    gotit = False
                    if not j_to_help_list: # if this list is actually empty stop
                        print('facility to help list is EMPTY -- heuristic stops here')
                        break
                    for j in j_to_help_list:
                        if gotit == True:
                            break
                        j_to_help = j[0][0]

                        # open the facility closest to j
                        closest_facility_list = get_closest_facility_list(j_to_help, F, dist)
                        open_list = [r_var for r_var, value in r.items() if value == 1]
                        for j in closest_facility_list:
                            if j[0] not in [elem[0] for elem in open_list]:
                                if (j_to_help,j[0]) not in trigger_2_tabu_list and B_used + c[j[0], 1] <= B:
                                    j_to_open = j[0]
                                    gotit = True
                                    break

                    if gotit == False:
                        print('All option explored -- heuristic stops here')
                        break

                    y[j_to_open] = 1

                    h,s = find_size_and_cluster_of_j(j_to_help, H, S, open_list)
                    r[j_to_open, 1, s] = 1
                    print('Trigger 2: found facility to open: is facility ' + str(j_to_open))
                    print('Trigger 2: found facility to help: is facility ' + str(j_to_help))


                # New Y and R vars are given to OP for a new solution:
                SP_opt_vars['y'] = y
                SP_opt_vars['r'] = r
                print('\n' + str(count) + 'th iteration of solving OP \n')
                OP_opt_vars_old = OP_opt_vars
                OP_opt_vars, OP_vars_list, OP_OptGap, OP_optval = OP_model(params, SP_opt_vars, gap_tol, OP_time_limit)
                OPs_OptGap_dict[count] = OP_OptGap
                OP_obj_evolution[count] = OP_optval

                SP_optval_k = evalute_SP_objval(OP_opt_vars,SP_opt_vars,params, gap_tol)
                SP_obj_evolution[count] = SP_optval_k

                if SP_optval_k < best_obj:
                    print('Heuristic iteration DID SUCCEED reducing obj val!')
                    best_obj = SP_optval_k

                    ######## leader vars ##################################
                    x_opt = pd.Series(SP_opt_vars['x']).reset_index()
                    x_opt.columns = ['i', 's', 'value']
                    x_opt.to_excel('x_optimal_vars_leader.xlsx')

                    r_opt = pd.Series(SP_opt_vars['r']).reset_index()
                    r_opt.columns = ['j', 'h', 's', 'value']
                    r_opt.to_excel('r_optimal_vars_leader.xlsx')
                    #######################################################

                    #### follower vars #####################################

                    z_opt = pd.Series(OP_opt_vars['z']).reset_index()
                    z_opt.columns = ['l', 'a', 'b', 'value']
                    z_opt.to_excel('z_optimal_vars_leader.xlsx')

                    p_opt = pd.Series(OP_opt_vars['p']).reset_index()
                    p_opt.columns = ['l', 'a', 'value']
                    p_opt.to_excel('p_optimal_vars_follower.xlsx')

                    v_opt = pd.Series(OP_opt_vars['v']).reset_index()
                    v_opt.columns = ['l', 'j', 'value']
                    v_opt.to_excel('v_optimal_vars_follower.xlsx')

                    ##################################################

                    best_k = count
                    # update open list
                    open_list = [r_var for r_var, value in r.items() if value == 1]
                    print('open list is: ')
                    print(open_list)
                else:
                    print('Heuristic iteration did not succeed in reducing obj val')
                    if trigger == 0:
                        trigger_0_tabu_list.append((j,h,s))
                        r[j, h + 1, s] = 0
                        r[j, h , s] = 1
                        B_used += c[j, h] - c[j, h + 1]
                        SP_opt_vars['y'] = y
                        SP_opt_vars['r'] = r
                        SP_opt_vars['n'] = n_before_attempt
                        OP_opt_vars = OP_opt_vars_old
                        open_list = [r_var for r_var, value in r.items() if value == 1]
                        print('open list is: ')
                        print(open_list)

                    if trigger == 1:
                        trigger_1_tabu_list.append((j_to_close,j_to_help,j_to_open))
                        y[j_to_close] = 1
                        r[j_to_close, h_to_close, s_to_close] = 1
                        y[j_to_open] = 0
                        r[j_to_open, h_to_close, s_to_close] = 0
                        SP_opt_vars['y'] = y
                        SP_opt_vars['r'] = r
                        SP_opt_vars['n'] = n_before_attempt
                        OP_opt_vars = OP_opt_vars_old
                        open_list = [r_var for r_var, value in r.items() if value == 1]
                        print('open list is: ')
                        print(open_list)

                    if trigger == 2:
                        trigger_2_tabu_list.append((j_to_help,j_to_open))
                        y[j_to_open] = 0
                        r[j_to_open, 1, s] = 0
                        SP_opt_vars['y'] = y
                        SP_opt_vars['r'] = r
                        SP_opt_vars['n'] = n_before_attempt
                        OP_opt_vars = OP_opt_vars_old
                        open_list = [r_var for r_var, value in r.items() if value == 1]
                        print('open list is: ')
                        print(open_list)

            ## hint for new heuristic develpoment
            # cluster definition may be inefficient --> attempt to change facility distribution across clusters

            if trigger == -1:
                print('facility to help list is EMPTY -- heuristic stops here')
                break

    # heuristic is ended
    perc_reduction = (first_eval - best_obj) / first_eval

    results = [instance_name, first_eval, best_obj, perc_reduction, maxit, NF, NC, ND, NV, SP_time_limit, OP_time_limit]
    for i in list(SP_obj_evolution.values()):
        results.append(i)

    OP_gap_results = []
    for gap in list(OPs_OptGap_dict.values()):
        OP_gap_results.append(gap)

    OP_obj_evolution_results = []
    for val in list(OP_obj_evolution.values()):
        OP_obj_evolution_results.append(val)

    print('trigger_0_tabu_list is: ' + str(trigger_0_tabu_list))
    print('trigger_1_tabu_list is: ' + str(trigger_1_tabu_list))
    print('trigger_2_tabu_list is: ' + str(trigger_2_tabu_list))

    return results, OP_gap_results, OP_obj_evolution_results

if __name__ == '__main__':

    test_realistic_inst = True
    if test_realistic_inst:

        instance_name = "inst_realistic"
        data = load_json_instance('./instances', instance_name + '.json')
        inst_data = data['inst_data']

        SP_time_limit = 30
        OP_time_limit = 1800
        maxit = 10

        results, OP_gap_results, OP_obj_evolution_results = heuristic(instance_name, maxit, SP_time_limit, OP_time_limit, test_realistic_inst)

    test_one_inst = False
    if test_one_inst:
        test_realistic_inst = False
        instance_num = 2  # 2 Heursitic Iteration n. 1: facility to help list is EMPTY -- heuristic stops here
        instance_name = 'inst_#' + str(instance_num)
        data = load_json_instance('./instances', instance_name + '.json')
        inst_data = data['inst_data']
        NC = inst_data['NC']

        if NC == 15:
            SP_time_limit = 25
            OP_time_limit = 200
        elif NC == 25:
            SP_time_limit = 35
            OP_time_limit = 400
        else:
            SP_time_limit = 50
            OP_time_limit = 800

        maxit = 10

        results, OP_gap_results, OP_obj_evolution_results = heuristic(instance_name, maxit, SP_time_limit, OP_time_limit, test_realistic_inst)


    test_all_inst = False
    if test_all_inst:
        test_realistic_inst = False
        results = []
        OP_gap_results = []
        OP_obj_results = []
        for instance_num in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:

            print('#########################\n INSTANCE NUMBER'+ str(instance_num)+'\n#################################')

            instance_name = 'inst_#' + str(instance_num)
            data = load_json_instance('./instances', instance_name + '.json')
            inst_data = data['inst_data']
            NC = inst_data['NC']

            if NC == 15:
                SP_time_limit = 25
                OP_time_limit = 200
            elif NC == 25:
                SP_time_limit = 35
                OP_time_limit = 400
            else:
                SP_time_limit = 50
                OP_time_limit = 800

            maxit = 10

            inst_results, inst_OP_gap_results, inst_OP_obj_evolution_results = heuristic(instance_name, maxit, SP_time_limit, OP_time_limit, test_realistic_inst)

            results.append(inst_results)
            OP_gap_results.append([instance_name]+inst_OP_gap_results)
            OP_obj_results.append([instance_name]+inst_OP_obj_evolution_results)

        iter_cols = [len(i) for i in results]
        iter_cols_len = max(iter_cols)
        itercols_name = ['first_eval'] + ['iter_#' + str(i) for i in range(1, iter_cols_len - 11)]
        df_columns = ['instance_name', 'first_eval', 'best_obj', 'perc_reduction', 'maxit', 'NF', 'NC', 'ND', 'NV', 'SP_time_limit', 'OP_time_limit'] + itercols_name
        df_results = pd.DataFrame(results, columns = df_columns)

        df_results.to_excel('heuristic_results_new.xlsx')

        OP_itergaps_name = ['gap_iter_#' + str(i) for i in range(1, iter_cols_len - 11 )]
        df_OP_gaps_columns = ['instance_name'] + ['first_eval_GAP'] + OP_itergaps_name
        df_OP_gaps = pd.DataFrame(OP_gap_results, columns = df_OP_gaps_columns)
        df_OP_gaps.to_excel('heuristic_OP_gaps_results.xlsx')

        OP_iterval_name = ['ObjVal_iter_#' + str(i) for i in range(1, iter_cols_len - 11 )]
        df_OP_ObjVal_columns = ['instance_name'] + ['first_eval_ObjVal'] + OP_iterval_name
        df_OP_ObjVal = pd.DataFrame(OP_obj_results, columns = df_OP_ObjVal_columns)
        df_OP_ObjVal.to_excel('heuristic_OP_ObjVal_results.xlsx')







