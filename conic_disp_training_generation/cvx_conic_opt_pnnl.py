############# Conic Programming Solver for Network Dispatch Optimization ######
###############################################################################
# The intention of this code is to setup a conic problem for optimizing the 
# unit commitment and dispatch of a network as exemplified by the Washington
# State University campus microgrid. 
#
# the script follows the sequence below:
# 1) system parameters are loaded
# 2) object classes are defined
# 3) constraint functions are defined
# 4) component variables are instantiated
# 5) objectiv function is created
# 6) problem constraint functions are collected
# 7) all variables and constraints are loaded to a problem instance
# 8) the problem definition is sent to the solver
# 9) the problem solution is sorted into desired dispatch format
#
# a microgrid parameter structure must be loaded which contains efficiency
# curves, limits, and inputs/outputs of all components, descriptions of all
# nodes, impedances of all lines, demand profiles of all nodes, and cost
# information about all resources
#
# This work is based off of the code which Kyle Monson at the Pacific
# Northwest National Lab wrote for the creation of a linear programming problem
# without a nodal network or power flow constraints.
#
# This code was written by Nadia Panossian at Washington State University and 
# was last updated on 10/17/2018 by Nadia Panossian
# the author can be reached at nadia.panossian@wsu.edu

import itertools

import os
import numpy as np
import pandas
import cvxpy
import pickle
from function.setup.piecewise_fit import piecewise_quadratic, piecewise_linear
import datetime
#import os

import time

from econ_dispatch.optimizer import get_conic_optimization_function



## ad user inputs 
v_nominal = 4135/4135
current_limit_value = 1.2
allow_dumping=True
bigM = 1e4 #cost of not meeting demand exactly   
c_mn = [1]#cooling line losses
h_mn = [1]#heat line losses
G = [0]#node admitance
B = []






# variable to indicate that we want all variables that match a pattern
# one item in the tuple key can be RANGE
RANGE = -1
#find_nodes creates a list of lists of indexes of equipment by node
# def find_nodes(comp_list):
#     comp_by_node = []
#     for node in network:
#         this_node = []
#         i = 0
#         for comp in comp_list:
#             check = any([True for equip in node.equipment if equip.name==comp.name])
#             if check:
#                 this_node.append(i)
#             i = i+1
#         comp_by_node.append(this_node)
#     return comp_by_node

####### POWER NETWORK OBJECT CLASS DEFINITIONS AND FUNCTIONS
class BuildAsset(object):
    def __init__(self, fundata, ramp_up=None, ramp_down=None, startcost=0, min_on=0, min_off=0, component_name=None):
        self.component_name = component_name
        self.fundata = {}
        for k, v in fundata.items():
            self.fundata[k] = np.array(v)
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.startcost = startcost
        self.min_on = min_on
        self.min_off = min_off

class BuildAsset_init(object):
    def __init__(self, status=0, output=0.0, command_history=[0]*24):
        self.status = status
        self.status1 = np.array(command_history)
        self.output = output

#  storage class
class Storage(object):
    def __init__(self, pmax, Emax, eta_ch, eta_disch, soc_max=1.0, soc_min=0.0, now_soc=0.0, component_name=None):
        self.pmax = pmax
        self.Emax = Emax
        self.eta_ch = eta_ch
        self.eta_disch = eta_disch
        self.soc_max = soc_max
        self.soc_min = soc_min
        self.now_soc = now_soc
        self.component_name = component_name
        if now_soc is None:
            raise ValueError("STATE OF CHARGE IS NONE")

def binary_var(var_name):
    return cvxpy.Variable(name=var_name, boolean=True)

def build_problem(forecast, parameters={}):
    #forecast is parasys, parameters are all the other csvs
    parasys = {}
    for fc in forecast:
        for key, value in fc.items():
            try: 
                parasys[key].append(value)
            except KeyError:
                parasys[key] = [value]
    
    fuel_cell_params = parameters.get("fuel_cell", {})
    microturbine_params = parameters.get("micro_turbine_generator", {})
    boiler_params = parameters.get("boiler", {})
    chiller_params = parameters.get("centrifugal_chiller_igv", {})
    abs_params = parameters.get("absorption_chiller", {})
    battery_params = parameters.get("battery", {})
    thermal_storage_params = parameters.get("thermal_storage", {})

    turbine_para = OrderedDict()
    turbine_init = OrderedDict()
    for name, parameters in itertools.chain(fuel_cell_params.items(), microturbine_params.items()):
        fundata = parameters["fundata"]
        ramp_up = parameters["ramp_up"]
        ramp_down = parameters["ramp_down"]
        start_cost = parameters["start_cost"]
        min_on = parameters["min_on"]
        output = parameters["output"]
        command_history = parameters["command_history"]
        turbine_para[name] = BuildAsset(fundata=fundata, 
                                        component_name= name,
                                        ramp_up = ramp_up,
                                        ramp_down = ramp_down,
                                        start_cost = start_cost,
                                        min_on = min_on)
        turbine_init[name] = BuildAsset_init(status=int(output>0.0), command_history = command_history, output=output)

    boiler_para = OrderedDict()
    boiler_init = OrderedDict()
    for name, parameters in boiler_params.items():
        fundata = parameters["fundata"]
        ramp_up = parameters["ramp_up"]
        ramp_down = parameters["ramp_down"]
        start_cost = parameters["start_cost"]
        output = parameters["output"]
        boiler_para[name] = BuildAsset(fundata=fundata, component_name=name, ramp_up=ramp_up, ramp_down=ramp_down, start_cost=start_cost)
        boiler_init[name] = BuildAsset_init(status=int(output>0.0), output=output)

    chiller_para = OrderedDict()
    chiller_init = OrderedDict()
    for name, parameters, in chiller_params.items():
        fundata = parameters["fundata"]
        ramp_up = parameters["ramp_up"]
        ramp_down = parameters["ramp_down"]
        start_cost = parameters["start_cost"]
        output = parameters["output"]
        chiller_para[name] = BuildAsset(fundata=fundata, component_name=name, ramp_up=ramp_up, ramp_down=ramp_down, start_cost=start_cost)
        chiller_init[name] = BuildAsset_init(status=int(output>0.0), output=output)

    abs_para = OrderedDict()
    abs_init = OrderedDict()
    for name, parameters, in abs_params.items():
        fundata = parameters["fundata"]
        ramp_up = parameters["ramp_up"]
        ramp_down = parameters["ramp_down"]
        start_cost = parameters["start_cost"]
        min_on = parameters["min_on"]
        min_off = parameters["min_off"]
        output = parameters["output"]
        command_history = parameters["command_history"]
        abs_para[name] = BuildAsset(fundata=fundata, component_name=name, ramp_up=ramp_up, ramp_down=ramp_down, start_cost=start_cost, min_on=min_on, min_off=min_off)# chiller1
        abs_init[name] = BuildAsset_init(status=int(output>0.0), command_history=command_history, output=output)

    e_storage_para = OrderedDict()
    # E_storage_para.append(Storage(Emax=2000.0, pmax=500.0, eta_ch=0.93, eta_disch=0.97, soc_min=0.1))
    for name, parameters in battery_params.items():
        e_storage_para[name] = Storage(Emax=parameters["cap"],
                                       pmax=parameters["max_power"],
                                       eta_ch=parameters["charge_eff"],
                                       eta_disch=parameters["discharge_eff"],
                                       soc_min=parameters["min_soc"],
                                       now_soc=parameters["soc"],
                                       component_name=name)
    c_storage_para = OrderedDict()
    h_storage_para = OrderedDict()
    for name, parameters in thermal_storage_params.items():
        if "col" in name:
            c_storage_para[name] = Storage(Emax=parameters["heat_cap"],
                                            pmax=parameters["max_power"],#5.0,
                                            eta_ch=parameters["eff"],#0.94,
                                            eta_disch=parameters["eff"],#0.94,
                                            now_soc=parameters["soc"],
                                            component_name=name)
        #otherwise its hot thermal
        else:
            h_storage_para[name] = Storage(Emax=parameters["heat_cap"],
                                            pmax=parameters["max_power"],#5.0,
                                            eta_ch=parameters["eff"],#0.94,
                                            eta_disch=parameters["eff"],#0.94,
                                            now_soc=parameters["soc"],
                                            component_name=name)


    a_hru = 0.8 #system heat loss factor = 1 - a_hru, 
    T = len(parasys["electricity_cost"])

    #create a list of lines by node
    e_lines_by_node = [] #list of connections by list of nodes
    c_lines_by_node = []
    h_lines_by_node = []
    i_e = 0 #index of line connection
    i_c = 0
    i_h = 0
    i = 0
    e_nodes = [0] #list of electrical nodes
    c_nodes = [0]
    h_nodes = [0]
    # for node in network:
    #     n_e = len(node.electrical.connections)
    #     n_c = len(node.district_cooling.connections)
    #     n_h = len(node.district_heat.connections)
    #     e_lines_by_node.append([i_e+i for i in range(n_e)])
    #     c_lines_by_node.append([i_c+i for i in range(n_c)])
    #     h_lines_by_node.append([i_h+i for i in range(n_h)])
    #     i_e += n_e
    #     i_c += n_c
    #     i_h += n_h
    #     if n_e>0:
    #         e_nodes.append(i)
    #     if n_c>0:
    #         c_nodes.append(i)
    #     if n_h>0:
    #         h_nodes.append(i)
    #     i +=1
    n_e_nodes = len(e_nodes) #number of electrical nodes
    n_c_nodes = len(c_nodes)
    n_h_nodes = len(h_nodes)
    n_e_lines = i_e
    n_c_lines = i_c
    n_h_lines = i_h


    n_turbines = len(turbine_para)
    n_dieselgen = len(diesel_para)
    n_boilers = len(boiler_para)
    n_chillers = len(chiller_para)
    n_abs = len(abs_para) # zero absorption chillers
    n_egrid = 1
    n_e_storage = len(e_storage_para)
    n_c_storage = len(c_storage_para)
    n_h_storage = len(h_storage_para)
    n_components = len(gen)
    n_nodes = len(network)
    states = []
    constraints = []
    date_range = [start_date+datetime.timedelta(hours=i) for i in range(T)]
    ## create list of components of each type at each node
    # grid_by_node = find_nodes(grid_para)
    # turbine_by_node = find_nodes(turbine_para)
    # diesel_by_node = find_nodes(diesel_para)
    # boiler_by_node = find_nodes(boiler_para)
    # chiller_by_node = find_nodes(chiller_para)
    # abs_by_node = find_nodes(abs_para)
    # e_storage_by_node = find_nodes(e_storage_para)
    # h_storage_by_node = find_nodes(h_storage_para)
    # c_storage_by_node = find_nodes(c_storage_para)
    # renew_by_node = find_nodes(renew_para)
    grid_by_node = [n for name in grid_params.items()]
    turbine_by_node = [n for name in itertools.chain(fuel_cell_params.items(), microturbine_params.items())]
    diesel_by_node = [n for name in diesel_params.items()]
    boiler_by_node = [n for name in boiler_params.items()]
    chiller_by_node = [n for name in chiller_params.items()]
    abs_by_node = [n for name in abs_params.items()]
    e_storage_by_node = [n for name in battery_params.items()]
    h_storage_by_node = [n for name in thermal_storage_params.items() if "col" not in name]
    c_storage_by_node = [n for name in thermal_storage_params.items() if "col" in name]
    #renew_by_node = [n for name in renewable_params.items()]




    #  storage functions
    def e_storage_pmax(index):
        i, t = index
        return e_storage_para[i].peak_disch * e_storage_para[i].size

    def e_storage_state_lower_bound(index):
        i, t = index
        return e_storage_para[i].max_dod #max depth of discharge

    def e_storage_state_upper_bound(index):
        i, t = index
        return e_storage_para[i].size

    def h_storge_pmax(index):
        i, t = index
        return h_storage_para[i].peak_disch * h_storage_para[i].size

    def h_storage_state_lower_bound(index):
        i, t = index
        return h_storage_para[i].max_dod #max depth of discharge

    def h_storage_state_upper_bound(index):
        i, t = index
        return h_storage_para[i].size

    def c_storage_pmax(index):
        i, t = index
        return c_storage_para[i].peak_disch * c_storage_para[i].size

    def c_storage_state_lower_bound(index):
        i, t = index
        return c_storage_para[i].max_dod

    def c_storage_state_upper_bound(index):
        i, t = index
        return c_storage_para[i].size

    def turbine_upper_bound(index):
        i, t = index
        return turbine_para[i].ub

    def turbine_lower_bound(index):
        i, t = index
        return turbine_para[i].lb

    def chiller_lower_bound(index):
        i, t = index
        return chiller_para[i].lb

    def chiller_upper_bound(index):
        i, t = index
        return chiller_para[i].ub

    def diesel_lower_bound(index):
        i, t = index
        return diesel_para[i].lb

    def diesel_upper_bound(index):
        i, t = index
        return diesel_para[i].ub

    def boiler_lower_bound(index):
        i, t = index
        return boiler_para[i].lb

    def boiler_upper_bound(index):
        i, t = index
        return boiler_para[i].ub




    #  all network objects create a group of variables associated with that object
    class VariableGroup(object):
        def __init__(self, name, indexes=(), is_binary_var=False, lower_bound_func=None, upper_bound_func=None, T=24, pieces=[1]):
            self.variables = {}

            name_base = name
            #if it is a piecewise function, make the variable group be a group of arrays (1,KK)
            if pieces==[1]:
                pieces = [1 for i in indexes[0]]

            #create name base string
            for _ in range(len(indexes)):
                name_base += "_{}"

            #create variable for each timestep and each component with a corresponding name
            for index in itertools.product(*indexes):
                var_name = name_base.format(*index)

                if is_binary_var:
                    var = binary_var(var_name)
                else:
                    #assign upper and lower bounds for the variable
                    if lower_bound_func is not None:
                        lower_bound = lower_bound_func(index)
                    else:
                        lower_bound = None

                    if upper_bound_func is not None:
                        upper_bound = upper_bound_func(index)
                    else:
                        upper_bound = None

                    #the lower bound should always be set if the upper bound is set
                    if lower_bound is None and upper_bound is not None:
                        raise RuntimeError("Lower bound should not be unset while upper bound is set")

                    #create the cp variable
                    if lower_bound_func == constant_zero:
                        var = cvxpy.Variable(pieces[index[0]], name = var_name, nonneg=True)
                    elif lower_bound is not None:
                        var = cvxpy.Variable(pieces[index[0]], name=var_name)
                        #constr = [var>=lower_bound]
                    elif upper_bound is not None:
                        var = cvxpy.Variable(pieces[index[0]],name=var_name)
                        #constr = [var<=upper_bound, var>=lower_bound]
                    else:
                        var = cvxpy.Variable(pieces[index[0]], name=var_name)
                    

                self.variables[index] = var
                #self.constraints[index] = constr
            
        #internal function to find variables associated with your key
        def match(self, key):
            position = key.index(RANGE)
            def predicate(xs, ys):
                z=0
                for i, (x, y) in enumerate(zip(xs, ys)):
                    if i != position and x==y:
                        z += 1
                return z == len(key)-1


            keys = list(self.variables.keys())
            keys = [k for k in keys if predicate(k,key)]
            keys.sort(key=lambda k: k[position]) 

            return [self.variables[k] for k in keys]

        #variable function to get the variables associated with the key
        def __getitem__(self, key):
            if type(key) != tuple:
                key = (key,)
                
            #n_ones = 0
            #for i, x in enumerate(key):
            #    if x == RANGE:
            #        n_ones+=1
            n_range = key.count(RANGE)

            if n_range == 0:
                return self.variables[key]
            elif n_range ==1:
                return self.match(key)
            else:
                raise ValueError("Can only get RANGE for one index.")


    def constant(x):
        def _constant(*args, **kwargs):
            return x
        return _constant

    constant_zero = constant(0)


    ######### CONSTRAINT FUNCTIONS
    def add_constraint(name, indexes, constraint_func):
        name_base = name
        for _ in range(len(indexes)):
            name_base +="_{}"

        for index in itertools.product(*indexes):
            name = name_base.format(*index)
            c = constraint_func(index)
            constraints.append((c,name))

    #balance equations
    #electric nodal balance
    def electric_p_balance(index):
        t, m = index
        n = [i for i in range(n_nodes) if network[i].name in network[m].electrical.connections]
        i_turb = turbine_by_node[m]
        i_grid = grid_by_node[m]
        i_chiller = chiller_by_node[m]
        i_es = e_storage_by_node[m]
        i_dies = diesel_by_node[m]
        i_lines = e_lines_by_node[m]
        #sum of power at node = Gmmxm + sum(Gmnymn+Bmnymn)
        #start with sum of power at node
        return cvxpy.sum([turbine_xp[j,t] for j in i_turb])\
        + cvxpy.sum([ep_elecfromgrid[j,t] - ep_electogrid[j,t] for j in i_grid])\
        - cvxpy.sum([chiller_yp[j,t] for j in i_chiller])\
        + cvxpy.sum([e_storage_dish[j,t] - e_storage_ch[j,t] for j in i_es])\
        + cvxpy.sum([dieselgen_xp[j,t] for j in i_dies])\
        - forecast.demand.ep[m,t]\
        + forecast.renew[m,t]\
        == G[m,m]*x_m[m,t]\
        + cvxpy.sum(G[m,n[RANGE]]*y_mn[i_lines[RANGE],t] + B[m,n[RANGE]]*z_mn[i_lines[RANGE],t])

    # reactive power nodal balance
    def electric_q_balance(index):
        t, m = index
        n = [i for i in range(n_nodes) if network[i].name in network[m].electrical.connections]
        i_turb = turbine_by_node[m]
        i_grid = grid_by_node[m]
        i_chiller = chiller_by_node[m]
        i_es = e_storage_by_node[m]
        i_dies = diesel_by_node[m]
        i_lines = e_lines_by_node[m]
        # sum of reactive power at node = -Bmmxm + sum(Gmnzmn - Bmnymn)
        # reative power at node is the sum of all power produced at that node minus any power consumed there
        # energy storage is assumed to store real power only, so not included here
        return cvxpy.sum([turbine_xq[j,t] for j in i_turb])\
        + cvxpy.sum([eq_elecfromgrid[j,t] - eq_electogrid[j,t] for j in i_grid])\
        - cvxpy.sum([chiller_yq[j,t] for j in i_chiller])\
        + cvxpy.sum([dieselgen_xq[j,t] for j in i_dies])\
        - forecast.demand.eq[m,t] ==\
        cvxpy.sum(G[m,n[RANGE]]*z_mn[i_lines[RANGE],t] - B[m,n[RANGE]]*y_mn[i_lines[RANGE],t]) -B[m,m]*x_m[m,t]

    #heat nodal balance
    def heat_balance(index):
        t, m = index
        n = [i for i in range(n_nodes) if network[i].name in network[m].district_heat.connections]
        i_turb = turbine_by_node[m]
        i_boiler = boiler_by_node[m]
        i_hs = h_storage_by_node[m]
        i_lines = h_lines_by_node[m]
        i_abs = abs_by_node[m]
        #sum of heat produced-heat used at this node = heat in/out of this node
        return cvxpy.sum([boiler_x[j,t] for j in i_boiler])\
        + cvxpy.sum([turbine_para[j].fundata["f_heat"]*turbine_xp[j,t] + turbine_para[j].fundata["c_heat"] for j in i_turb])\
        + cvxpy.sum([h_storage_disch[j,t] - h_storage_ch[j,t] for j in i_hs])\
        - cvxpy.sum([abs_y[j,t] for j in i_abs])\
        - forecast.demand.h[m,t]\
        - heat_dump[m,t]\
        + heat_unserve[m,t]\
        - cvxpy.sum([heat_loss[m,n[j]]*h_mn[i_lines[j],t] for j in range(len(n))])\
        == 0

    # cooling power nodal balance
    def cool_balance(index):
        t, m = index
        n = [i for i in range(n_nodes) if network[i].name in network[m].district_cooling.connections]
        i_chiller = chiller_by_node[m]
        i_abs = abs_by_node[m]
        i_cs = c_storage_by_node[m]
        i_lines = c_lines_by_node[m]
        return cvxpy.sum([abs_x[j,t] for j in i_abs]) + cvxpy.sum([chiller_x[j,t] for j in i_chiller])\
        + cvxpy.sum([c_storage_disch[j,t] - c_storage_ch[j,t] for j in i_cs])\
        - cool_dump[m,t]\
        + cool_unserve[m,t]\
        - forecast.demand.c[m,t] ==\
        cvxpy.sum([cool_loss[m,n[j]]*c_mn[i_lines[j],t] for j in range(len(n))])

    # voltage constraints
    def voltage_limit_upper(index):
        t, m = index
        return (v_nominal*.9)**2 <= x_m[m,t] 

    def voltage_limit_lower(index):
        t, m = index
        return x_m[m,t] <= (v_nominal*1.1)**2

    # line current limits
    def current_limit(index):
        t, m = index
        n = [i for i in range(n_nodes) if network[i].name in network[m].electrical.connections]
        i_lines = e_lines_by_node[m]
        return (G[m,m]**2 + B[m,n[RANGE]]**2)*(x_m[m,t] + x_m[n[RANGE],t] - 2*y_mn[i_lines[RANGE],t])<= current_limit_value**2

    # equality to assure that x,y,z variable subsitution holds: xmxn = ymn^2 + zmn^2
    def electric_interrelation(index):
        t, m = index #m is the node
        n = [i for i in range(n_nodes) if network[i].name in network[m].electrical.connections]
        i_lines = e_lines_by_node[m]
        return cvxpy.power(x_m[n[RANGE],t] + x_m[m,t],2) - cvxpy.power(x_m[n[RANGE],t], 2) - cvxpy.power(x_m[m,t], 2) >= cvxpy.power(y_mn[i_lines[RANGE],t], 2) + cvxpy.power(z_mn[i_lines[RANGE], t], 2) 
        #return x_m[n[RANGE],t]*x_m[m,t] == y_mn[i_lines[RANGE],t]**2 + z_mn[i_lines[RANGE],t]**2

    # equality to assure that cooling leaving one node is recorded negative from one node and positive for the other
    def line_heat(index):
        t, m = index #m is the node
        m_line = h_lines_by_node[m]
        n = [i for i in range(n_nodes) if network[i].name in network[m].district_heat.connections]
        n_line = [h_lines_by_node[i][network[i].district_heat.connections.index(network[m].name)] for i in n]#index of line from n to m 
        return h_mn[m_line[RANGE],t] == -h_mn[n_line[RANGE],t]

    def line_cooling(index):
        t, m = index #m is the node
        m_line = c_lines_by_node[m]#index of line from m to n
        n = [i for i in range(n_nodes) if network[i].name in network[m].district_cooling.connections]
        n_line = [c_lines_by_node[i][network[i].district_cooling.connections.index(network[m].name)] for i in n]#index of line from n to m 
        return c_mn[m_line[RANGE],t] == -c_mn[n_line[RANGE],t]

    # storage constraint functions

    def e_storage_state_constraint(index):
        i, t = index
        return e_storage_state[i,t] == e_storage_state[i,t-1] + e_storage_para[i].eta_ch * e_storage_ch[i,t] - 1/e_storage_para[i].eta_disch * e_storage_disch[i,t]

    def e_storage_init(index):
        i = index[0]
        return e_storage_state[i,0] == e_storage0[i] + e_storage_para[i].eta_ch * e_storage_ch[i,0] - 1/e_storage_para[i].eta_disch * e_storage_disch[i,0]

    def h_storage_state_constraint(index):
        i, t = index
        return h_storage_state[i,t] == h_storage_state[i,t-1] + h_storage_para[i].eta_ch * h_storage_ch[i,t] - 1/h_storage_para[i].eta_disch * h_storage_disch[i,t]

    def h_storage_init(index):
        i = index[0]
        return h_storage_state[i,0] == h_storage0[i] + h_storage_para[i].eta_ch * h_storage_ch[i,0] - 1/h_storage_para[i].eta_disch * h_storage_disch[i,0]

    def c_storage_init(index):
        i = index[0]
        return c_storage_state[i,1] == c_storage0[i] + c_storage_para[i].charge_eff * c_storage_ch[i,1] - 1/c_storage_para[i].disch_eff * c_storage_disch[i,1]

    def c_storage_state_constraint(index):
        i, t = index
        return c_storage_state[i,1] == c_storage_state[i,t-1] + c_storage_para[i].charge_eff * c_storage_ch[i,t] - 1/c_storage_para[i].disch_eff * c_storage_disch[i,t]

    # turbnie constraint functions
    #this constraint is stated as (bp*x + cip)^2 - ep*x - d - y <= 0
    # the cost of y will drive it to be equal to (bp*x + cip)^2 - ep*x -d
    def turbine_y_consume(index):
        i, t = index
        return cvxpy.norm(turbine_para[i].fundata["bp"]*turbine_xp_k[i,t],2)\
        + turbine_para[i].fundata["fp"]*turbine_xp_k[i,t]\
        + turbine_para[i].fundata["cp"]*turbine_s_k[i,t]\
        + cvxpy.norm(turbine_para[i].fundata["bq"]*turbine_xq_k[i,t],2)\
        + turbine_para[i].fundata["fq"]*turbine_xq_k[i,t]\
        + turbine_para[i].fundata["cq"]*turbine_s_k[i,t]\
        - turbine_y[i,t] <= 0
        # return cvxpy.norm(turbine_para[i].fundata["bp"]*turbine_xp_k[i,t] + turbine_para[i].fundata["cip"],2)\
        # + turbine_para[i].fundata["ep"]*turbine_xp_k[i,t]\
        # + turbine_para[i].fundata["dp"]\
        # + cvxpy.norm(turbine_para[i].fundata["bq"]*turbine_xq_k[i,t] + turbine_para[i].fundata["ciq"],2)\
        # + turbine_para[i].fundata["eq"]*turbine_xq_k[i,t]\
        # + turbine_para[i].fundata["dq"]\
        # - turbine_y[i,t] <= 0 

    def turbine_xp_generate(index):
        i, t = index
        return turbine_xp[i,t] == cvxpy.sum(turbine_xp_k[i,t])

    def turbine_xq_generate(index):
        i, t = index
        return turbine_xq[i,t] == cvxpy.sum(turbine_xq_k[i,t])

    def turbine_xp_k_lower(index):
        i, t = index
        #individual lower bounds are non-zero
        return turbine_para[i].lb*turbine_s_k[i,t] <= turbine_xp_k[i,t]

    def turbine_xp_k_upper(index):
        i, t = index
        return turbine_para[i].ub**2*turbine_s_k[i,t] >= cvxpy.power(turbine_xp_k[i,t], 2) + cvxpy.power(turbine_xq_k[i,t],2)

    def turbine_x_status(index):
        i, t = index
        #return turbine_s[i,t] == cvxpy.sum(turbine_s_k[i,t])
        return 1>=cvxpy.sum(turbine_s_k[i,t])

    # def turbine_start_status1(index):
    #     i, t = index[0], 0
    #     return turbine_start[i,t] >= turbine_s[i,1] - init_turbine[i].status

    def turbine_start_status(index):
        i, t = index
        return turbine_start[i,t] >= turbine_s[i,t] - turbine_s[i,t-1]

    def turbine_ramp1_up(index):
        i, t = index[0], 0
        return turbine_init[i] + turbine_para[i].ramp_rate <= turbine_xp[i,t]

    def turbine_ramp1_down(index):
        i, t = index[0], 0
        return turbine_xp[i,t] <= turbine_init[i] + turbine_para[i].ramp_rate

    def turbine_ramp_up(index):
        i, t = index
        return turbine_xp[i,t-1] + turbine_para[i].ramp_rate >= turbine_xp[i,t]

    def turbine_ramp_down(index):
        i, t = index
        return turbine_xp[i,t-1] - turbine_para[i].ramp_rate <= turbine_xp[i,t]

    #turbines lock on time limit not defined in this configuration

    #diesel generator constraint functions

    def dieselgen_y_consume(index):
        i, t = index
        return cvxpy.norm(diesel_para[i].fundata["bp"]*dieselgen_xp_k[i,t],2)\
        + diesel_para[i].fundata["fp"]*dieselgen_xp_k[i,t]\
        - diesel_para[i].fundata["cp"]*dieselgen_s_k[i,t]\
        + cvxpy.norm(diesel_para[i].fundata["bq"]*dieselgen_xq_k[i,t],2)\
        + diesel_para[i].fundata["fq"]*dieselgen_xq_k[i,t]\
        - diesel_para[i].fundata["cq"]*dieselgen_s_k[i,t]\
        - dieselgen_y[i,t] <= 0

    def dieselgen_xp_generate(index):
        i, t = index
        return dieselgen_xp[i,t] == cvxpy.sum(dieselgen_xp_k[i,t])

    def dieselgen_xq_generate(index):
        i, t = index
        return dieselgen_xq[i,t] == cvxpy.sum(dieselgen_xq_k[i,t])

    def dieselgen_xp_k_lower(index):
        i, t = index
        #individual lower bounds are zero
        return diesel_para[i].lb*dieselgen_s_k[i,t] <= dieselgen_xp_k[i,t]

    def dieselgen_xp_k_upper(index):
        i, t = index
        return diesel_para[i].ub*dieselgen_s_k[i,t] >= dieselgen_xp_k[i,t]

    def dieselgen_xp_lower(index):
        i, t = index
        #individual lower bounds are zero
        return diesel_para[i].lb * dieselgen_s[i,t] <= dieselgen_xp[i,t]

    def dieselgen_xp_upper(index):
        i, t = index
        return diesel_para[i].ub * dieselgen_s_k[i,t] >= dieselgen_xp_k[i,t]

    def dieselgen_x_status(index):
        i, t = index
        #return dieselgen_s[i,t]== cvxpy.sum(dieselgen_s_k[i,t])
        return 1>=cvxpy.sum(dieselgen_s_k[i,t])

    def dieselgen_start_status(index):
        i, t = index
        return dieselgen_start[i,t] >= dieselgen_s[i,t] - dieselgen_s[i,t-1]

    def dieselgen_ramp1_up(index):
        i, t = index[0], 0
        return dieselgen_xp[i,t] <= dieselgen_init[i] + diesel_para[i].ramp_rate

    def dieselgen_ramp1_down(index):
        i, t = index[0], 0
        return dieselgen_init[i] + diesel_para[i].ramp_rate <= dieselgen_xp[i,t]

    def dieselgen_ramp_up(index):
        i, t = index
        return dieselgen_xp[i,t-1] + diesel_para[i].ramp_rate >= dieselgen_xp[i,t]

    def dieselgen_ramp_down(index):
        i, t = index
        return dieselgen_xp[i,t-1] - diesel_para[i].ramp_rate <= dieselgen_xp[i,t]


    # boiler constraint functions

    def boiler_y_consume(index):
        i, t = index 
        return cvxpy.norm(boiler_para[i].fundata["b"]*boiler_x_k[i,t], 2)\
        + boiler_para[i].fundata["f"]*boiler_x_k[i,t]\
        + boiler_para[i].fundata["d"]*boiler_s_k[i,t] <=0
        # return cvxpy.norm(boiler_para[i].fundata["b"]*boiler_x_k[i,t] + boiler_para[i].fundata["c"], 2)\
        # + boiler_para[i].fundata["e"]*boiler_x_k[i,t]\
        # + boiler_para[i].fundata["d"] - boiler_y[i,t] <= 0

    def boiler_x_generate(index):
        i, t = index
        return boiler_x[i,t] == cvxpy.sum(boiler_x_k[i,t])

    def boiler_x_k_lower(index):
        i, t = index
        return boiler_s_k[i,t]*boiler_para[i].lb <= boiler_x_k[i,t]

    def boiler_x_k_upper(index):
        i, t = index
        return boiler_para[i].ub*boiler_s_k[i,t] >= boiler_x_k[i,t]

    def boiler_x_lower(index):
        i, t = index
        return boiler_para[i].lb * boiler_s[i,t] <= boiler_x[i,t]

    def boiler_x_upper(index):
        i, t = index
        return boiler_para[i].ub * boiler_s_k[i,t] >= boiler_x_k[i,t]

    def boiler_x_status(index):
        i, t = index
        #return boiler_s[i,t] == cvxpy.sum(boiler_s_k[i,t])
        return 1>=cvxpy.sum(boiler_s_k[i,t])

    def boiler_start_status1(index):
        i, t = index[0], 0
        return boiler_start[i,t] >= boiler_s[i,1] - boiler_init[i].status

    def boiler_start_status(index):
        i, t = index
        return boiler_start[i,t] >= boiler_s[i,t] - boiler_s[i,t-1]

    def boiler_ramp1_up(index):
        i, t = index[0], 0
        return boiler_x[i,t] <= boiler_init[i] + boiler_para[i].ramp_rate
    def boiler_ramp1_down(index):
        i, t = index[0], 0
        return boiler_init[i] - boiler_para[i].ramp_rate <= boiler_x[i,t]

    def boiler_ramp_up(index):
        i, t = index
        return boiler_x[i,t-1] + boiler_para[i].ramp_rate >= boiler_x[i,t]

    def boiler_ramp_down(index):
        i, t = index
        return boiler_x[i,t-1] - boiler_para[i].ramp_rate <= boiler_x[i,t]

    # chiller constraints

    def chiller_yp_consume(index):
        i, t = index
        return cvxpy.norm(chiller_para[i].fundata["bp"]*chiller_x_k[i,t], 2)\
        + chiller_para[i].fundata["fp"]*chiller_x_k[i,t]\
        + chiller_para[i].fundata["cp"]*chiller_s_k[i,t]\
        -chiller_yp[i,t] <= 0
        # return cvxpy.norm(chiller_para[i].fundata["bp"]*chiller_x_k[i,t] + chiller_para[i].fundata["cip"], 2)\
        # + chiller_para[i].fundata["ep"]*chiller_x_k[i,t]\
        # + chiller_para[i].fundata["dp"]*chiller_s_k[i,t]\
        # - chiller_yp[i,t] <= 0

    def chiller_yq_consume(index):
        i, t = index
        return cvxpy.norm(chiller_para[i].fundata["bq"]*chiller_x_k[i,t], 2)\
        + chiller_para[i].fundata["fq"]*chiller_x_k[i,t]\
        + chiller_para[i].fundata["cq"]*chiller_s_k[i,t] - chiller_yq[i,t] <= 0

    def chiller_x_generate(index):
        i, t = index
        return chiller_x[i,t] == cvxpy.sum(chiller_x_k[i,t])

    def chiller_x_lower(index):
        i, t = index
        return chiller_para[i].lb * chiller_s[i,t] <= chiller_x[i,t]

    def chiller_x_k_lower(index):
        i, t = index
        return chiller_para[i].lb*chiller_s_k[i,t] <= chiller_x_k[i,t]

    def chiller_x_k_upper(index):
        i, t = index
        return chiller_para[i].ub * chiller_s_k[i,t] >= chiller_x_k[i,t]

    def chiller_x_status(index):
        i, t = index
        #return chiller_s[i,t] == cvxpy.sum(chiller_s_k[i,t])
        return 1>=cvxpy.sum(chiller_s_k[i,t])

    def chiller_start_status1(index):
        i, t = index[0], 0
        return chiller_start[i,t] >= chiller_s[i,1] - chiller_init[i].status

    def chiller_start_status(index):
        i, t = index
        return chiller_start[i,t] >= chiller_s[i,t] - chiller_s[i, t-1]

    def chiller_ramp1_up(index):
        i, t = index[0], 0
        return chiller_x[i,t] <= chiller_init[i] + chiller_para[i].ramp_rate

    def chiller_ramp1_down(index):
        i, t = index[0], 0
        return chiller_init[i] - chiller_para[i].ramp_rate <= chiller_x[i,t]

    def chiller_ramp_up(index):
        i, t = index
        return chiller_x[i, t-1] + chiller_para[i].ramp_rate >= chiller_x[i,t]

    def chiller_ramp_down(index):
        i, t = index
        return chiller_x[i, t-1] - chiller_para[i].ramp_rate <= chiller_x[i,t]

    # absorption chiller constraints

    def abs_y_consume(index):
        i, t = index
        return abs_y[i,t] == cvxpy.sum(abs_para[i].fundata["h"] * np.power(abs_x_k[i,t,RANGE],2) + abs_para[i].fundata["f"] * abs_x_k[i,t,RANGE] + abs_para[i].fundata["c"]*abs_s_k[i,t,RANGE])

    def abs_x_generate(index):
        i, t = index
        return abs_x[i,t] == cvxpy.sum(abs_x_k[i,t,RANGE])

    def abs_x_lower(index):
        i, t = index
        return abs_para[i].lb * abs_s[i,t] <= abs_x[i,t]

    def abs_x_k_lower(index):
        i, t = index
        return abs_s_k[i,t]*abs_para[i].lb <= abs_x_k[i,t]

    def abs_x_k_upper(index):
        i, t = index
        return abs_para[i].ub * abs_s_k[i,t] >= abs_x_k[i,t]

    def abs_x_status(index):
        i, t = index
        #return abs_s[i,t] == cvxpy.sum(abs_s_k[i,t])
        return 1>=cvxpy.sum(abs_s_k[i,t])

    def abs_start_status(index):
        i, t = index
        return abs_start[i,t] >= abs_s[i,t] - abs_s[i, t-1]

    def abs_ramp1_up(index):
        i, t = index[0], 0
        return abs_x[i,t] <= abs_init[i] + abs_para[i].ramp_rate

    def abs_ramp1_down(index):
        i, t = index[0], 0
        return abs_init[i] - abs_para[i].ramp_rate <= abs_x[i,t]

    def abs_ramp_up(index):
        i, t = index
        return abs_x[i,t-1] + abs_para[i].ramp_rate >= abs_x[i,t]

    def abs_ramp_down(index):
        i, t = index
        return abs_x[i,t-1] - abs_para[i].ramp_rate <= abs_x[i,t]

    # power dumping constraint
    def wasted_heat(index):
        t = index[0]
        return q_hru_heating_in[0,t] == turbine_y[0,t] - turbine_xp[0,t]/293.1 + turbine_y[1,t] - turbine_xp[1,t]/293.1 #-abs_y[0,t]

    def hru_limit(index):
        t = index[0]
        return q_hru_heating_out[0,t] <= a_hru * q_hru_heating_in[0,t]

    toc = time.time()-tic
    print('load all data and functions' + str(toc))

    ###### DEFINE COMPONENT TYPES
    # electric grid
    tic = time.time()
    index_hour = (range(T),)
    index_nodes = range(n_nodes), range(T)
    ep_elecfromgrid = VariableGroup("ep_elecfromgrid", indexes=index_nodes, lower_bound_func=constant_zero) #real power from grid
    eq_elecfromgrid = VariableGroup("eq_elecfromgrid", indexes=index_nodes, lower_bound_func=constant_zero) #reactive power from grid
    ep_electogrid = VariableGroup("ep_electogrid", indexes=index_nodes, lower_bound_func=constant_zero) #real power to the grid
    eq_electogrid = VariableGroup("eq_electogrid", indexes=index_nodes, lower_bound_func=constant_zero) #reactive power from grid

    #dumping allowance
    if allow_dumping:
        if n_boilers>0:
            heat_unserve = VariableGroup("heat_unserve", indexes=index_nodes, lower_bound_func=constant_zero)
            heat_dump = VariableGroup("heat_dump", indexes=index_nodes, lower_bound_func=constant_zero)
        if n_chillers>0:
            cool_unserve = VariableGroup("cool_unserve", indexes=index_nodes, lower_bound_func=constant_zero)
            cool_dump = VariableGroup("cool_dump", indexes=index_nodes, lower_bound_func=constant_zero)

    #turbines: # fuel cells are considered turbines
    index_turbines = range(n_turbines), range(T)
    turbine_y = VariableGroup("turbine_y", indexes =index_turbines, lower_bound_func=constant_zero) #  fuel use
    turbine_xp = VariableGroup("turbine_xp", indexes=index_turbines, lower_bound_func=constant_zero)  #  real power output
    turbine_xq = VariableGroup("turbine_xq", indexes=index_turbines, lower_bound_func=constant_zero)  #  reactive power output
    turbine_xp_k = VariableGroup("turbine_xp_k", indexes=index_turbines, pieces=turbine_pieces) #  power outputs from all piecewise parts
    turbine_xq_k = VariableGroup("turbine_xq_k", indexes=index_turbines,pieces=turbine_pieces) #  power outputs from all piecewise parts
    turbine_s_k = VariableGroup("turbine_s_k", indexes=index_turbines, is_binary_var=True, pieces=turbine_pieces) #  states from all pieceswise parts
    #turbine_s = VariableGroup("turbine_s", indexes=index_turbines, is_binary_var=True)# unit commitment of turbine
    #turbine_start = VariableGroup("turbine_start", indexes=index_turbines, is_binary_var=True) #  is the turbine starting up

    #diesel generators:
    index_dieselgen = range(n_dieselgen), range(T)
    dieselgen_y = VariableGroup("dieselgen_y", indexes=index_dieselgen, lower_bound_func=constant_zero) #fuel use
    dieselgen_xp = VariableGroup("dieselgen_xp", indexes=index_dieselgen, lower_bound_func=constant_zero) # real power output
    dieselgen_xq = VariableGroup("dieselgen_xq", indexes=index_dieselgen, lower_bound_func=constant_zero) # reactive power output
    dieselgen_xp_k = VariableGroup("dieselgen_xp_k", indexes=index_dieselgen, pieces=diesel_pieces) # power outputs from all piecewise parts
    dieselgen_xq_k = VariableGroup("dieselgen_xq_k", indexes=index_dieselgen, pieces=diesel_pieces) # power outputs from all piecewise parts
    dieselgen_s_k = VariableGroup("dieselgen_s_k", indexes=index_dieselgen, is_binary_var=True, pieces=diesel_pieces) # states from all piecewise pats
    #dieselgen_s = VariableGroup("dieselgen_s", indexes=index_dieselgen, is_binary_var = True) #unit commitment
    #dieselgen_start = VariableGroup("dieselgen_start", indexes=index_dieselgen, is_binary_var=True) # is the turbine starting up


    #boilers:
    index_boilers = range(n_boilers), range(T)
    boiler_y = VariableGroup("boiler_y", indexes=index_boilers, lower_bound_func=constant_zero) #  fuel use from boiler
    boiler_x = VariableGroup("bioler_x", indexes=index_boilers, lower_bound_func=constant_zero) #  heat output from boiler
    boiler_x_k = VariableGroup("boiler_x_k", indexes=index_boilers, pieces=boiler_pieces) #  heat output from each portion of the piecewise fit
    boiler_s_k = VariableGroup("boiler_s_k", indexes=index_boilers, is_binary_var=True, pieces=boiler_pieces) #  unit commitment for each portion of the piecewise efficiency fit
    #boiler_s = VariableGroup("boiler_s", indexes = index_boilers, is_binary_var = True) #unit commitment
    #boiler_start = VariableGroup("boiler_start", indexes=index_boilers, is_binary_var=True) #  is the boiler starting up

    #chillers
    index_chiller = range(n_chillers), range(T)
    chiller_x = VariableGroup("chiller_x", indexes = index_chiller, lower_bound_func = constant_zero) #  cooling power output
    chiller_yp = VariableGroup("chiller_yp", indexes = index_chiller, lower_bound_func = constant_zero) #  real electric power demand
    chiller_yq = VariableGroup("chiller_yq", indexes = index_chiller, lower_bound_func = constant_zero) #  reactive electric power demand
    chiller_x_k = VariableGroup("chiller_x_k", indexes = index_chiller, pieces=chiller_pieces) #  cooling output from all piecewise parts
    chiller_s_k = VariableGroup("chiller_s_k", indexes=index_chiller, is_binary_var = True, pieces=chiller_pieces) #  unit commitment for piecewise sections
    #chiller_s = VariableGroup("chiller_s", indexes=index_chiller, is_binary_var=True) #unit commitment
    #chiller_start = VariableGroup("chiller_start", indexes=index_chiller, is_binary_var=True) #  is the chiller starting up

    #absorption chillers
    index_abs = range(n_abs), range(T)
    abs_x = VariableGroup("abs_x", indexes = index_abs, lower_bound_func = constant_zero) #  cooling power output
    abs_y = VariableGroup("abs_y", indexes = index_abs, lower_bound_func = constant_zero) #  heat power demand
    abs_x_k = VariableGroup("abs_x_k", indexes = index_abs, pieces=abs_pieces) #  cooling output from all piecewise parts
    abs_s_k = VariableGroup("abs_s_k", indexes=index_abs, is_binary_var = True, pieces=abs_pieces) #  unit commitment for piecewise sections
    #abs_s = VariableGroup("abs_s", indexes = index_abs, is_binary_var = True) #unit commitment
    #abs_start = VariableGroup("abs_start", indexes=index_abs, is_binary_var=True) #  is the chiller starting up

    #storage
    #electric storage
    index_e_storage = range(n_e_storage), range(T)
    e_storage_disch = VariableGroup("e_storage_disch", indexes=index_e_storage, lower_bound_func = constant_zero, upper_bound_func = e_storage_pmax)
    e_storage_ch = VariableGroup("e_storage_ch", indexes = index_e_storage, lower_bound_func = constant_zero, upper_bound_func = e_storage_pmax)
    e_storage_state = VariableGroup("e_storage_state", indexes = index_e_storage, lower_bound_func =e_storage_state_lower_bound, upper_bound_func = e_storage_state_upper_bound)
    #cold water tank or other cold energy storage
    index_c_storage = range(n_c_storage), range(T)
    c_storage_disch = VariableGroup("c_storage_disch", indexes = index_c_storage, lower_bound_func = constant_zero, upper_bound_func = c_storage_pmax)
    c_storage_ch = VariableGroup("c_storage_ch", indexes = index_c_storage, lower_bound_func = constant_zero, upper_bound_func = c_storage_pmax)
    c_storage_state = VariableGroup("c_storage_state", indexes = index_c_storage, lower_bound_func = c_storage_state_lower_bound, upper_bound_func = c_storage_state_upper_bound)
    #hot water tank or other hot energy storage
    if n_h_storage>0:
        index_h_storage = range(n_h_storage), range(T)
        h_storage_disch = VariableGroup("h_storage_disch", indexes=index_h_storage, lower_bound_func=constant_zero, upper_bound_func = h_storage_pmax)
        h_storage_ch = VariableGroup("h_storage_ch", indexes = index_h_storage, lower_bound_func = constant_zero, upper_bound_func = h_storage_pmax)
        h_storage_state = VariableGroup("h_storage_state", indexes = index_h_storage, lower_bound_func = h_storage_state_lower_bound, upper_bound_func = h_storage_state_upper_bound)

    #nodal network
    #voltage is split into x, y, z
    #x_m = v_m^2 and is therefore positive
    #y_mn = v_m*v_n*cos(theta_mn)
    #z_mn = v_m*v_n*sin(theta_mn)
    index_e_nodes = range(n_e_nodes), range(T)
    index_e_lines = range(n_e_lines), range(T)
    x_m = VariableGroup("x_m", indexes = index_e_nodes, lower_bound_func = constant_zero)
    y_mn = VariableGroup("y_mn", indexes = index_e_lines) 
    z_mn = VariableGroup("z_mn", indexes = index_e_lines)
    #heat network
    index_h_nodes = range(n_h_nodes), range(T)
    index_h_lines = range(n_h_lines), range(T)
    h_mn = VariableGroup("h_mn", indexes = index_h_lines, lower_bound_func = constant_zero)
    #cooling network
    index_c_lines = range(n_c_lines), range(T)
    c_mn = VariableGroup("c_mn", indexes = index_c_lines, lower_bound_func = constant_zero)

    #define utility costs
    pelec_cost = parasys["electricity_cost"]
    qelec_cost = np.divide(pelec_cost,5)
    pselback_rate = np.divide(pelec_cost,-2)
    qselback_rate = -qelec_cost
    gas_rate = parasys["natural_gas_cost"]
    diesel_rate = parasys["lambda_diesel"]

    #forecast generation and demand
    forecast = TestData()
    forecast.demand.h = np.zeros((len(network),T))
    forecast.demand.c = np.zeros((len(network),T))
    setattr(forecast.demand, 'ep', np.zeros((len(network), T)))
    setattr(forecast.demand, 'eq', np.zeros((len(network), T)))
    setattr(forecast, 'renew', np.zeros((len(network),T)))
    i = 0
    for node in network:
        ep_demand = parasys["electric_load"][t]
        forecast.demand.ep[i,:] = ep_demand
        forecast.demand.eq[i,:] = np.multiply(ep_demand, 0.1)#assume high power factor for now
        forecast.demand.h[i,:] = parasys["heat_load"]
        forecast.demand.c[i,:] = parasys["cool_load"]
        forecast.renew[i,:] = parasys["solar_kW"]
        i +=1


    toc = time.time()-tic
    print('Variables '+str(toc))

    ######## OBJECTIVE FUNCTION

    tic = time.time()
    objective_components = []

    for var, _lambda in zip(ep_elecfromgrid[RANGE], pelec_cost):
        objective_components.append(var * _lambda)

    for var, _lambda in zip(eq_elecfromgrid[RANGE], qelec_cost):
        objective_components.append(var * _lambda)

    for var, _lambda in zip(ep_electogrid[(RANGE,)], pselback_rate):
        objective_components.append(var * _lambda)

    for var, _lambda in zip(eq_electogrid[(RANGE,)], qselback_rate):
        objective_components.append(var * _lambda)

    for i in range(n_turbines):
        for var, state, _lambda in zip(turbine_y[i, RANGE], turbine_s_k[i,RANGE], gas_rate):
            objective_components.append(var * _lambda)# * sum(state))
        # for var, _lambda in zip(turbine_s_k[i,RANGE], gas_rate):
        #     objective_components.append((turbine_para[i].fundata["dp"]*cvxpy.sum(var))*_lambda)
        # for var in turbine_start[i, RANGE]:
        #     objective_components.append(var * turbine_para[i].start_cost)

    for i in range(n_dieselgen):
        for var, _lambda in zip(dieselgen_y[(i, RANGE)], diesel_rate):
            objective_components.append(var * _lambda)
        # for var, _lambda in zip(dieselgen_s_k[i,RANGE], diesel_rate):
        #     objective_components.append((diesel_para[i].fundata["dp"]*cvxpy.sum(var))*_lambda)
        # for var in dieselgen_start[i, RANGE]:
        #     objective_components.append(var * diesel_para[i].start_cost)

    for i in range(n_boilers):
        for var, _lambda in zip(boiler_y[i, RANGE], gas_rate):
            objective_components.append(var * _lambda)
        # for var, _lambda in zip(boiler_s_k[i,RANGE], gas_rate):
        #     objective_components.append((boiler_para[i].fundata["dh"]*cvxpy.sum(var))*_lambda)
        # for var in boiler_start[i, RANGE]:
        #     objective_components.append(var * boiler_para[i].start_cost)

    #only penalize unserved demand
    for group in (heat_unserve, cool_unserve):
        for var in group[RANGE]:
            objective_components.append(var * bigM)

    toc = time.time()-tic
    print('objective function '+str(toc))

    ######## ADD CONSTRAINTS
    tic = time.time()
    index_without_first_hour = (range(1,T),)

    # add turbine constraints 
    index_turbine = (range(n_turbines),)
    add_constraint("turbine_y_consume", index_turbine + index_hour, turbine_y_consume) #False
    add_constraint("turbine_xp_generate", index_turbine + index_hour, turbine_xp_generate) #True
    add_constraint("turbine_xp_k_lower", index_turbine + index_hour, turbine_xp_k_lower)
    add_constraint("turbine_xp_k_upper", index_turbine + index_hour, turbine_xp_k_upper)
    add_constraint("turbine_x_status", index_turbine + index_hour, turbine_x_status)
    #add_constraint("turbine_start_status1", index_turbine, turbine_start_status1)
    #add_constraint("turbine_start_status", index_turbine + index_without_first_hour, turbine_start_status)
    add_constraint("turbine_ramp1_up", index_turbine, turbine_ramp1_up)
    add_constraint("turbine_ramp1_down", index_turbine, turbine_ramp1_down)
    add_constraint("turbine_ramp_up", index_turbine + index_without_first_hour, turbine_ramp_up)
    add_constraint("turbine_ramp_down", index_turbine + index_without_first_hour, turbine_ramp_down)
    #add_constraint("turbines_lock_on1", index_turbine, turbines_lock_on1)

    # add diesel constraints
    index_diesel = (range(n_dieselgen),)
    add_constraint("dieselgen_y_consume", index_diesel + index_hour, dieselgen_y_consume)
    add_constraint("dieselgen_xp_generator", index_diesel + index_hour, dieselgen_xp_generate)
    add_constraint("dieselgen_xp_k_lower", index_diesel + index_hour, dieselgen_xp_k_lower)
    add_constraint("dieselgen_xp_k_upper", index_diesel + index_hour, dieselgen_xp_k_upper)
    add_constraint("dieselgen_x_status", index_diesel + index_hour, dieselgen_x_status)
    #add_constraint("dieselgen_start_status", index_diesel + index_without_first_hour, dieselgen_start_status)
    add_constraint("dieselgen_ramp1_up", index_diesel, dieselgen_ramp1_up)
    add_constraint("dieselgen_ramp1_down", index_diesel, dieselgen_ramp1_down)
    add_constraint("dieselgen_ramp_up", index_diesel + index_without_first_hour, dieselgen_ramp_up)
    add_constraint("dieselgen_ramp_down", index_diesel + index_without_first_hour, dieselgen_ramp_down)

    # add boiler constraints
    index_boiler = (range(n_boilers),)
    add_constraint("boiler_y_consume", index_boiler + index_hour, boiler_y_consume)
    add_constraint("boiler_x_generate", index_boiler + index_hour, boiler_x_generate)
    add_constraint("boiler_x_k_lower", index_boiler + index_hour, boiler_x_k_lower)
    add_constraint("boiler_x_k_upper", index_boiler + index_hour, boiler_x_k_upper)
    add_constraint("boiler_x_status", index_boiler + index_hour, boiler_x_status)
    #add_constraint("boiler_start_status1", index_boiler + index_hour, boiler_start_status1)
    #add_constraint("boiler_start_status", index_boiler + index_without_first_hour, boiler_start_status)
    add_constraint("boiler_ramp1_up", index_boiler, boiler_ramp1_up)
    add_constraint("boiler_ramp1_down", index_boiler, boiler_ramp1_down)
    add_constraint("boiler_ramp_up", index_boiler + index_without_first_hour, boiler_ramp_up)
    add_constraint("boiler_ramp_down", index_boiler + index_without_first_hour, boiler_ramp_down)

    #add chiller constriants
    index_chiller = (range(n_chillers),)
    add_constraint("chiller_yp_consume", index_chiller + index_hour, chiller_yp_consume)
    add_constraint("chiller_yq_consume", index_chiller + index_hour, chiller_yq_consume)
    add_constraint("chiller_x_generate", index_chiller + index_hour, chiller_x_generate)
    add_constraint("chiller_x_k_lower", index_chiller + index_hour, chiller_x_k_lower)
    add_constraint("chiller_x_k_upper", index_chiller + index_hour, chiller_x_k_upper)
    add_constraint("chiller_x_status", index_chiller + index_hour, chiller_x_status)
    #add_constraint("chiller_start_status1", index_chiller, chiller_start_status1)
    #add_constraint("chiller_start_status", index_chiller + index_without_first_hour, chiller_start_status)
    add_constraint("chiller_ramp1_up", index_chiller, chiller_ramp1_up)
    add_constraint("chiller_ramp1_down", index_chiller, chiller_ramp1_down)
    add_constraint("chiller_ramp_up", index_chiller + index_without_first_hour, chiller_ramp_up)
    add_constraint("chiller_ramp_down", index_chiller + index_without_first_hour, chiller_ramp_down)

    #add absorption chillers
    index_abs = (range(n_abs,),)
    add_constraint("abs_y_consume", index_abs + index_hour, abs_y_consume)
    add_constraint("abs_x_generate", index_abs + index_hour, abs_x_generate)
    add_constraint("abs_x_lower", index_abs + index_hour, abs_x_lower)
    add_constraint("abs_x_k_lower", index_abs + index_hour, abs_x_k_lower)
    add_constraint("abs_x_k_upper", index_abs + index_hour, abs_x_k_upper)
    add_constraint("abs_x_status", index_abs + index_hour, abs_x_status)
    #add_constraint("abs_start_status1", index_abs, abs_start_status1)
    #add_constraint("abs_start_status", index_abs + index_without_first_hour, abs_start_status)
    add_constraint("abs_ramp1_up", index_abs, abs_ramp1_up)
    add_constraint("abs_ramp1_down", index_abs, abs_ramp1_down)
    add_constraint("abs_ramp_up", index_abs + index_without_first_hour, abs_ramp_up)
    add_constraint("abs_ramp_down", index_abs + index_without_first_hour, abs_ramp_down)

    #wasted heat
    #add_constraint("wasted_heat", index_hour, wasted_heat)
    #add_constraint("HRUlimit", index_hour, HRUlimit)


    # add storage constraints
    index_e_storage = (range(n_e_storage),)
    e_storage0 = np.zeros(n_e_storage)
    for i in range(n_e_storage):
        e_storage0[i] = 0.5*e_storage_para[i].size #default is to start storage at 50%
    add_constraint("e_storage_init", index_e_storage, e_storage_init)
    add_constraint("e_storage_state_constraint", index_e_storage + index_without_first_hour, e_storage_state_constraint)

    index_h_storage = (range(n_h_storage),)
    h_storage0 = np.zeros(n_h_storage)
    for i in range(n_h_storage):
        h_storage0[i] = 0.5*h_storage_para[i].size #default is to start storage at 50%
    add_constraint("h_storage_init", index_h_storage, h_storage_init)
    add_constraint("h_storage_state_constraint", index_h_storage + index_without_first_hour, h_storage_state_constraint)

    index_c_storage = (range(n_c_storage),)
    c_storage0 = np.zeros(n_c_storage)
    for i in range(n_c_storage):
        c_storage0[i] = 0.5 * c_storage_para[i].size #default is to start storage at 50% energy
    add_constraint("c_storage_init", index_c_storage, c_storage_init)
    add_constraint("c_storage_state_constraint", index_c_storage + index_without_first_hour, c_storage_state_constraint)


    # add equality constraints for supply and demand
    #for m in range(n_nodes):
    m = (range(n_nodes),)
    add_constraint("electric_p_balance", index_hour + m, electric_p_balance)
    add_constraint("electric_q_balance", index_hour + m, electric_q_balance)
    add_constraint("cool_balance", index_hour + m, cool_balance)
    add_constraint("heat_balance", index_hour + m, heat_balance)

    # add line and voltage limits
    #for m in range(n_nodes):
    add_constraint("voltage_limit_upper", index_hour + m, voltage_limit_upper)
    add_constraint("voltage_limit_lower", index_hour + m, voltage_limit_lower)
    add_constraint("current_limit", index_hour + m, current_limit)

    # add variable subsitution constraint
    #for m in range(n_nodes):
    #add_constraint("electric_interrelation", index_hour + m, electric_interrelation) # not sure this constraint is needed
    add_constraint("line_heat", (range(T), tuple(h_nodes)), line_heat)
    add_constraint("line_cooling", (range(T), tuple(c_nodes)), line_cooling)

    toc = time.time()-tic
    print('add constraints: '+ str(toc))

    print('problem parameters loaded')

    ######## SOLVE FINAL PROBLEM
    objective = cvxpy.Minimize(cvxpy.sum(objective_components))
    constraints_list = [x[0] for x in constraints]
    prob = cvxpy.Problem(objective, constraints_list)
    print('problem created, solving problem')

    # tic = time.time()
    # result = prob.solve(verbose = True, solver='ECOS_BB')#solver = ECOS

    # toc = time.time()-tic
    # print('optimal cost: '+ str(result))
    # print(prob._solution)
    # print('problem solved in '+str(toc)+'seconds')
    #print(x_m.value)

    return prob

######## SORT PROBLEM SOLUTION


def get_conic_optimization_function(config):
    return get_pulp_optimization_function(build_problem, config)
