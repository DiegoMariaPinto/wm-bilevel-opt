import itertools
from gurobipy import *
from gurobipy import quicksum
from gurobipy import GRB
from gurobipy import Model

import pandas as pd
import numpy as np
from Get_instances import load_json_instance






def heuristic(instance_name, maxit, SP_time_limit, OP_time_limit):

    data = load_json_instance('./instances', instance_name + '.json')
    inst_data = data['inst_data']
    disdur = data['disdur_dict']

    gap_tol = 0.0025  # 1e-5

    NF = inst_data['NF']
    NC = inst_data['NC']
    ND = inst_data['ND']
    NV = inst_data['NV']


    results = [instance_name] # list of results info to be appended

    return results

if __name__ == '__main__':

    instance_name = 'inst_#' + str(1)
    maxit = 3
    SP_time_limit = 20
    OP_time_limit = 120

    results = heuristic(instance_name, maxit, SP_time_limit, OP_time_limit)

    ########################