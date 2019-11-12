############# Conic Programming Solver for Network Dispatch Optimization ######
###############################################################################
# The intention of this code is to setup a conic problem for optimizing the 
# unit commitment and dispatch of a network as exemplified by
# Example Problem 7.4 in Power Systems Analysis by Hadi Saadat
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
import pandas as pd
import openpyxl
import csv
from coreapi import codecs
from copy import copy
#import xlsxwriter
import cvxpy
import pickle
from class_definition.plant_struct import Plant, Network, Optimoptions
from class_definition.test_data import TestData
from class_definition.component import (ElectricChiller, AbsorptionChiller, CombinedHeatPower, ElectricGenerator, Heater)
from class_definition.component import (ElectricStorage, ThermalStorage, Utility, Renewable)
from function.setup.piecewise_fit import piecewise_quadratic, piecewise_linear
import datetime
from function.setup.update_qpform_all import (load_piecewise, fit_coproduction, remove_segments, fit_fcn)
from pickle_wsu_campus_demand import load_demand
#import os

import time




########## READ IN SYSTEM PARAMETERS
tic = time.time()
start_date = datetime.datetime(2010, 1, 1, 0, 0, 0)



#read in forecast, gen, network
with open(os.getcwd() + '\\library\\demo_7_9.pickle', 'rb') as file_object:
    plant = pickle.load(file_object)

gen = plant.generator
network = plant.network
optimoptions = plant.optimoptions


#test_data = load_demand()
# with open(os.getcwd() + '\\library\\data\\wsu_campus_demand_2009_2012', 'rb') as file_object:
#     test_data = pickle.load(file_object, encoding='latin1')

## ad user inputs 
v_base = 4.135 # nominal voltage on lines kV
p_base = 100 # power base value kVA
current_base = p_base/v_base 
r_base = v_base**2/p_base # kohms nominal value to normalize admitance (kv^2/kVA = kohms)
j_base = r_base
v_nominal = 1.0#4135/4135
current_limit_value = 290/current_base # amps limit on 250 kcmil wire between 170 and 290
voltage_deviation = .1 # max normalized deviation of voltage
T = 1 #horizon
timesteps = 1 #number of receding horizon repetitions
allow_dumping= False#True
allow_thermal_slack = False
bigM = 10#1e2 #cost of not meeting demand exactly
grid_limit = 100

#functions to process information from generator list and network description
## this library sorts components by type and by type by node
# sort components into their own lists and count types
turbine_para = []

grid_para = []
n_utility = 0
n_gas_util = 0
e_storage0 = []
KK =1 # number of piecwise sections per component efficiency curves
turbine_pieces = []
#abs_init = np.zeros((n_abs,1))
var_name_list = []

for i in range(len(gen)):
    if isinstance(gen[i], ElectricGenerator):
        gen[i].size = gen[i].size/p_base
        gen[i].ramp_rate = gen[i].ramp_rate/p_base
        #create efficiency piecewise quadratic fit curve
        #fit_terms, x_min, x_max = piecewise_quadratic(gen[i].output.capacity, gen[i].output.electricity, error_thresh=0.1, resolution=1, max_cap=gen[i].size)
        x_min = [0]
        x_max = [gen[i].size]
        turbine_para.append(gen[i])
        turbine_pieces.append(len(x_max))

    


#find_nodes creates a list of lists of indexes of equipment by node
def find_nodes(comp_list):
    comp_by_node = []
    for node in network:
        this_node = []
        i = 0
        for comp in comp_list:
            check = any([True for equip in node.equipment if equip.name==comp.name])
            if check:
                this_node.append(i)
            i = i+1
        comp_by_node.append(this_node)
    return comp_by_node

#create a list of lines by node
e_lines_by_node = [] #list of connections by list of nodes
i_e = 0 #index of line connections
i = 0
e_nodes = [] #list of electrical nodes
e_connected_nodes = []
for node in network:
    n_e = len(node.electrical.connections)
    e_lines_by_node.append([i_e+i for i in range(n_e)])
    i_e += n_e
    if n_e>0:
        e_connected_nodes.append(i)
    if n_e>0 or node.electrical.load !=None:
        e_nodes.append(i)
    i +=1
n_e_nodes = len(e_nodes) #number of electrical nodes
n_e_connected_nodes = len(e_connected_nodes) #number of electrical nodes that are connected to other nodes
n_e_lines = i_e


Y = np.zeros((5,5), dtype = np.complex)
Y[0,1] = -1/(0.02+0.06j)
Y[1,0] = Y[0,1]
Y[0,2] = -1/(0.08+0.24j)
Y[2,0] = Y[0,2]
Y[1,2] = -1/(0.06+0.18j)
Y[2,1] = Y[1,2]
Y[1,3] = -1/(0.06 + 0.18j)
Y[3,1] = Y[1,3]
Y[1,4] = -1/(0.04+0.12j)
Y[4,1] = Y[1,4]
Y[2,3] = -1/(0.01+0.03j)
Y[3,2] = Y[2,3]
Y[3,4] = -1/(0.08 + 0.24j)
Y[4,3] = Y[3,4]

Y[0,0] = -sum(Y[0,:]) +0.030j +0.025j
Y[1,1] = -sum(Y[1,:]) +0.030j +0.020j +0.020j +0.015j
Y[2,2] = -sum(Y[2,:]) +0.025j +0.020j +0.010j
Y[3,3] = -sum(Y[3,:]) +0.020j +0.010j +0.025j
Y[4,4] = -sum(Y[4,:]) +0.015j +0.025j

G = Y.real
B = Y.imag

# count components
n_turbines = len(turbine_para)
#n_abs = len(abs_para) # zero absorption chillers
n_egrid = 1
n_components = len(gen)
n_nodes = len(network)
states = []
constraints = []
date_range = [start_date+datetime.timedelta(hours=i) for i in range(T)]
## create list of components of each type at each node
turbine_by_node = find_nodes(turbine_para)

# set the starting value for x_n
# it is used as an upper limit, so allow it to be the upper bound on voltage
x_n = np.ones((n_e_nodes,T))*(1+voltage_deviation)**2
# x_n[0,0] = 1.06**2+0.1
# x_n[1,0] = 1.045**2+0.1
# x_n[2,0] = 1.03**2+0.1


# variable to indicate that we want all variables that match a pattern
# one item in the tuple key can be RANGE
RANGE = -1


####### POWER NETWORK OBJECT CLASS DEFINITIONS AND FUNCTIONS
class BuildAsset(object):
    def __init__(self, fundata, ramp_up=None, ramp_down=None, startcost=0, min_on=0, min_off=0, component_name=None):
        self.component_name = component_name
        self.fundata = fundata
        self.ramp_up = ramp_up
        self.ramp_down = ramp_down
        self.startcost = startcost
        self.min_on = min_on
        self.min_off = min_off

class BuildAsset_init(object):
    def __init__(self, status=0, output=0.0):
        self.status = status
        self.status1 = np.zeros(T)
        self.output = output

def binary_var(var_name):
    return cvxpy.Variable(name=var_name, boolean=True)



#gas utility pricing function
def find_gas_pricing(date_stamp):
    i=0
    day_stamp = datetime.datetime(year=date_stamp.year, month=date_stamp.month, day=date_stamp.day)
    price_ind = fuel_para[i].timestamp.index(day_stamp)
    gas_rate = fuel_para[i].rate[price_ind] # price in $/thousand cubic feet
    gas_rate = gas_rate*(1/293.07) # convert to $/kWh gas --> 293.07 kWh/thousand cubic feet natural gas
    return gas_rate


#  all network objects create a group of variables associated with that object
class VariableGroup(object):
    def __init__(self, name, indexes=(), is_binary_var=False, lower_bound_func=None, upper_bound_func=None, T=T, pieces=[1]):
        global var_name_list
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
            var_name_list.append(var_name)
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

toc = time.time()-tic
print('load all data and functions' + str(toc))

def run_horizon(timestep, v_iters, x_n, pid_error_last):
    # INPUTS:
    # timestep: the timestamp for the timesteps in the horizon
    # v_iters: an integer denoting how many times you have tried to converge on this problem
    # x_n: last iterations attempt at voltage values
    # pid_error: iteration error used for pid convergence
    #
    # OUTPUTS:
    # v_iters: integer denoting either number of iterations past, or 0 indicating convergence
    # pid_error: error in voltage values as applied to the PID convergence
    # x_n: the iteration's values for voltage at nodes
    # 
    #balance equations
    #electric nodal balance
    def electric_p_balance(index):
        t, m = index
        n = [i for i in range(n_nodes) if network[i].name in network[m].electrical.connections]
        i_turb = turbine_by_node[m]
        i_lines = e_lines_by_node[m]
        #sum of power at node = Gmmxm + sum(Gmnymn+Bmnymn)
        #start with sum of power at node
        if not n== []:
            return cvxpy.sum([turbine_xp[j,t] for j in i_turb])\
            - forecast.demand.ep[m,t]\
            == G[m,m]*x_m[m,t]\
            + cvxpy.sum(G[m,n[RANGE]]*y_mn[i_lines[RANGE],t] + B[m,n[RANGE]]*z_mn[i_lines[RANGE],t])
        else:
            return cvxpy.sum([turbine_xp[j,t] for j in i_turb])\
            - forecast.demand.ep[m,t]\
            == G[m,m]*x_m[m,t]
        #normalized by 3000

    # reactive power nodal balance
    def electric_q_balance(index):
        t, m = index
        n = [i for i in range(n_nodes) if network[i].name in network[m].electrical.connections]
        i_turb = turbine_by_node[m]
        i_lines = e_lines_by_node[m]
        # sum of reactive power at node = -Bmmxm + sum(Gmnzmn - Bmnymn)
        # reative power at node is the sum of all power produced at that node minus any power consumed there
        # energy storage is assumed to store real power only, so not included here
        if not n==[]:
            return cvxpy.sum([turbine_xq[j,t] for j in i_turb])\
            - forecast.demand.eq[m,t] + B[m,m]*x_m[m,t]\
            == cvxpy.sum(G[m,n[RANGE]]*z_mn[i_lines[RANGE],t] - B[m,n[RANGE]]*y_mn[i_lines[RANGE],t])
        else:
            return cvxpy.sum([turbine_xq[j,t] for j in i_turb])\
            - forecast.demand.eq[m,t]\
            == -B[m,m]*x_m[m,t]

    # voltage constraints
    def voltage_limit_lower(index):
        t, m = index
        return (v_nominal*(1.0-voltage_deviation))**2 <= x_m[m,t] 

    def voltage_limit_upper(index):
        t, m = index
        return x_m[m,t] <= (v_nominal*(voltage_deviation+1.0))**2

    def y_mn_equality(index):
        t, m = index
        # find lines out
        m_lines = e_lines_by_node[m]
        # find lines back in
        node_ns = [i for i in range(n_nodes) if network[i].name in network[m].electrical.connections]
        n_lines = []
        for n in node_ns:
            this_line = [i for i in range(len(e_lines_by_node[n])) if network[m].name == network[n].electrical.connections[i]]
            n_lines.append(e_lines_by_node[n][this_line[0]])
        # set those lines equal for y_mn
        return y_mn[m_lines[RANGE],t] == y_mn[n_lines[RANGE],t]

    def z_mn_inequality(index):
        t, m = index
        # find lines out
        m_lines = e_lines_by_node[m]
        # find lines back in
        node_ns = [i for i in range(n_nodes) if network[i].name in network[m].electrical.connections]
        n_lines = []
        for n in node_ns:
            this_line = [i for i in range(len(e_lines_by_node[n])) if network[m].name == network[n].electrical.connections[i]]
            n_lines.append(e_lines_by_node[n][this_line[0]])
        # set those lines equal to the negative of each other for z_mn
        return z_mn[m_lines[RANGE],t] == -z_mn[n_lines[RANGE],t]


    # def y_voltage_limit_lower(index):
    #     t, m = index
    #     return 0.8*(v_nominal*0.9)**2 <= y_mn[m,t]

    # def y_voltage_limit_upper(index):
    #     t, m = index 
    #     return y_mn[m,t] <= (v_nominal*0.1)**2

    # def z_voltage_limit_upper(index):
    #     t, m = index
    #     return z_mn[m,t] <= (v_nominal*100)**2

    # def z_voltage_limit_lower(index):
    #     t, m = index
    #     return -0.2*(v_nominal*1.1)**2 <= z_mn[m,t]


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
        #return cvxpy.power(x_m[n[RANGE],t] + x_m[m,t],2) - cvxpy.power(x_m[n[RANGE],t], 2) - cvxpy.power(x_m[m,t], 2) >= cvxpy.power(y_mn[i_lines[RANGE],t], 2) + cvxpy.power(z_mn[i_lines[RANGE], t], 2) 
        #return x_m[n[RANGE],t]*x_m[m,t] == y_mn[i_lines[RANGE],t]**2 + z_mn[i_lines[RANGE],t]**2
        #return -1/2*cvxpy.square(x_m[m,t] + x_m[n[RANGE],t])+v_nominal**2 >= cvxpy.square(y_mn[i_lines[RANGE],t]) + cvxpy.square(z_mn[i_lines[RANGE],t])
        #return x_m[m,t]*x_m[n[0],t] >= y_mn[i_lines[0],t]**2 + z_mn[i_lines[0],t]**2
        
        return x_m[m,t]*x_n[n[RANGE],t] >= (y_mn[i_lines[RANGE],t]**2 + z_mn[i_lines[RANGE],t]**2)
        #use binomial approximation
        #return 2*x_m[m,t]-1 >=y_mn[i_lines[RANGE],t]**2 + z_mn[i_lines[RANGE],t]**2

    def known_voltages(index):
        t, m = index
        return x_m[m,t] == x_n[m,t]


    # turbnie constraint functions
    # (hx)^2 + fx +c
    #this constraint is stated as (bp*x + cip)^2 - ep*x - d - y <= 0
    # the cost of y will drive it to be equal to (bp*x + cip)^2 - ep*x -d
    def turbine_y_consume(index):
        i, t = index
        return turbine_para[i].fundata["bp"]*cvxpy.power(p_base*turbine_xp_k[i,t],2)\
        + turbine_para[i].fundata["fp"]*turbine_xp_k[i,t]*p_base\
        - turbine_y[i,t] <= 0
    # + turbine_para[i].fundata["cp"]\

    def turbine_xp_generate(index):
        i, t = index
        return turbine_xp[i,t] == cvxpy.sum(turbine_xp_k[i,t])

    def turbine_xq_generate(index):
        i, t = index
        return turbine_xq[i,t] == cvxpy.sum(turbine_xq_k[i,t])

    def turbine_xp_k_lower(index):
        i, t = index
        #individual lower bounds are non-zero
        return (turbine_para[i].lb) <= turbine_xp_k[i,t]

    def turbine_xq_k_lower(index):
        i, t = index
        return (turbine_para[i].lb)*0.01 <= turbine_xq_k[i,t]

    def turbine_xp_k_upper(index):
        i, t = index
        return (turbine_para[i].ub) >= turbine_xp_k[i,t] 
        #+ cvxpy.power(turbine_xq_k[i,t],2)

    def turbine_x_status(index):
        i, t = index
        #return turbine_s[i,t] == cvxpy.sum(turbine_s_k[i,t])
        return 1==cvxpy.sum(turbine_s_k[i,t])

    def turbine_powerfactor_limit_upper(index):
        i, t = index
        return turbine_xq[i,t] <= turbine_xp[i,t]*2

    def turbine_powerfactor_limit_lower(index):
        i, t = index
        return turbine_xq[i,t] >= -turbine_xp[i,t]*0.2

    #turbines lock on time limit not defined in this configuration

    def no_slack_e(index):
        i, t = index
        return elec_unserve[i,t] == 0

    ###### DEFINE COMPONENT TYPES

    tic = time.time()
    #dumping allowance
    #elec_unserve = VariableGroup("elec_unserve", indexes=index_nodes, lower_bound_func=constant_zero)
    #if allow_thermal_slack==False:
        #add_constraint("no_slack_e", index_nodes, no_slack_e)


    #turbines: # fuel cells are considered turbines
    if n_turbines>0:
        index_turbines = range(n_turbines), range(T)
        turbine_y = VariableGroup("turbine_y", indexes =index_turbines, lower_bound_func=constant_zero) #  fuel use
        turbine_xp = VariableGroup("turbine_xp", indexes=index_turbines, lower_bound_func=constant_zero)  #  real power output
        turbine_xq = VariableGroup("turbine_xq", indexes=index_turbines, lower_bound_func=constant_zero)  #  reactive power output
        turbine_xp_k = VariableGroup("turbine_xp_k", indexes=index_turbines, pieces=turbine_pieces, lower_bound_func=constant_zero) #  power outputs from all piecewise parts
        turbine_xq_k = VariableGroup("turbine_xq_k", indexes=index_turbines,pieces=turbine_pieces) #  power outputs from all piecewise parts
        # turbine_s_k = VariableGroup("turbine_s_k", indexes=index_turbines, is_binary_var=True, pieces=turbine_pieces) #  states from all pieceswise parts
    #turbine_s = VariableGroup("turbine_s", indexes=index_turbines, is_binary_var=True)# unit commitment of turbine



    #voltage is split into x, y, z
    #x_m = v_m^2 and is therefore positive
    #y_mn = v_m*v_n*cos(theta_mn)
    #z_mn = v_m*v_n*sin(theta_mn)
    index_e_nodes = range(n_e_nodes), range(T)
    index_e_lines = range(n_e_lines), range(T)
    x_m = VariableGroup("x_m", indexes = index_e_nodes, lower_bound_func = constant_zero)
    y_mn = VariableGroup("y_mn", indexes = index_e_lines, lower_bound_func = constant_zero) 
    z_mn = VariableGroup("z_mn", indexes = index_e_lines)
    #heat network

    #define utility costs
    if n_utility>0:
        pelec_cost = [find_utility_pricing(date_stamp) for date_stamp in date_range]
        qelec_cost = np.multiply(pelec_cost,5)
    else:
        pelec_cost = 0
        qelec_cost = 0
    pselback_rate = np.multiply(pelec_cost,0)
    qselback_rate = np.multiply(qelec_cost, 0)
    if n_gas_util>0:
        gas_rate = [find_gas_pricing(date_stamp) for date_stamp in date_range]
    #diesel_rate = [find_diesel_pricing(date_stamp) for date_stamp in date_range]

    #forecast generation and demand
    forecast = TestData()
    setattr(forecast.demand, 'ep', np.zeros((len(network), T)))
    setattr(forecast.demand, 'eq', np.zeros((len(network), T)))
    i = 0
    for node in network:
        #if not node.electrical.load == []:
        #ep_demand = [find_demand(date_stamp, 'e', n=node.electrical.load) for date_stamp in date_range]
        forecast.demand.ep[i,:] = node.electrical.load[0]/p_base #np.multiply(ep_demand, 1/n_nodes)
        forecast.demand.eq[i,:] = node.electrical.load[1]/p_base#assume high power factor for now
        i +=1


    toc = time.time()-tic
    print('Variables '+str(toc))

    ######## OBJECTIVE FUNCTION

    tic = time.time()
    objective_components = []

    gas_rate = [1]
    for i in range(n_turbines):
        for var, _lambda in zip(turbine_y[i, RANGE], gas_rate):
            objective_components.append(var * _lambda)



    #only penalize unserved demand
    # if allow_thermal_slack:
    #     for group in (heat_unserve, cool_unserve, elec_unserve):
    #         for var in group[RANGE]:
    #             objective_components.append(var * bigM)

    #toc = time.time()-tic
    #print('objective function '+str(toc))

    ######## ADD CONSTRAINTS
    #tic = time.time()
    index_without_first_hour = (range(1,T),)
    index_hour = (range(T),)

    # add equality constraints for supply and demand
    #for m in range(n_nodes):
    m_index = (e_nodes,)
    known_index = (range(2),)
    add_constraint("electric_p_balance", index_hour + m_index, electric_p_balance)
    add_constraint("electric_q_balance", index_hour + m_index, electric_q_balance)
    #add_constraint("known_voltages", index_hour+known_index, known_voltages)

    m_connected_index = (e_connected_nodes,)
    # add_constraint("current_limit", index_hour + m_connected_index, current_limit)
    add_constraint("y_mn_equality", index_hour + m_connected_index, y_mn_equality)
    # add_constraint("z_mn_inequality", index_hour + m_connected_index, z_mn_inequality)

    # add variable subsitution constraint
    #for m in range(n_nodes):
    # add_constraint("electric_interrelation", index_hour + m_connected_index, electric_interrelation) # not sure this constraint is needed

    # add turbine constraints 
    index_turbine = (range(n_turbines),)
    add_constraint("turbine_y_consume", index_turbine + index_hour, turbine_y_consume) #False
    add_constraint("turbine_xp_generate", index_turbine + index_hour, turbine_xp_generate) #True
    add_constraint("turbine_xq_generate", index_turbine + index_hour, turbine_xq_generate)
    add_constraint("turbine_xp_k_lower", index_turbine + index_hour, turbine_xp_k_lower)
    add_constraint("turbine_xp_k_upper", index_turbine + index_hour, turbine_xp_k_upper)
    # add_constraint("turbine_x_status", index_turbine + index_hour, turbine_x_status)
    #add_constraint("turbine_start_status1", index_turbine, turbine_start_status1)
    #add_constraint("turbine_start_status", index_turbine + index_without_first_hour, turbine_start_status)
    #add_constraint("turbine_ramp1_up", index_turbine, turbine_ramp1_up)
    #add_constraint("turbine_ramp1_down", index_turbine, turbine_ramp1_down)
    #add_constraint("turbine_ramp_up", index_turbine + index_without_first_hour, turbine_ramp_up)
    #add_constraint("turbine_ramp_down", index_turbine + index_without_first_hour, turbine_ramp_down)
    # add_constraint("turbine_powerfactor_limit_upper", index_turbine+index_hour, turbine_powerfactor_limit_upper)
    # add_constraint("turbine_powerfactor_limit_lower", index_turbine+index_hour, turbine_powerfactor_limit_lower)
    #add_constraint("turbines_lock_on1", index_turbine, turbines_lock_on1)

    # add line and voltage limits
    #for m in range(n_nodes):
    add_constraint("voltage_limit_upper", index_hour + m_index, voltage_limit_upper)
    add_constraint("voltage_limit_lower", index_hour + m_index, voltage_limit_lower)
    # add_constraint("y_voltage_limit_upper", index_hour+ n_index, y_voltage_limit_upper)
    # add_constraint("y_voltage_limit_lower", index_hour+n_index, y_voltage_limit_lower)
    # add_constraint("z_voltage_limit_upper", index_hour+n_index, z_voltage_limit_upper)
    # add_constraint("z_voltage_limit_lower", index_hour+n_index, z_voltage_limit_lower)#this one makes it switch from primal unbounded, to primal infeasible



    #toc = time.time()-tic
    #print('add constraints: '+ str(toc))

    print('problem parameters loaded')


    ######## SOLVE FINAL PROBLEM
    objective = cvxpy.Minimize(cvxpy.sum(objective_components))
    constraints_list = [x[0] for x in constraints]
    prob = cvxpy.Problem(objective, constraints_list)
    print('problem created, solving problem')

    tic = time.time()
    result = prob.solve(solver='GUROBI',verbose=True)#, FeasibilityTol=1e-2, BarConvTol=1)#, verbose=True)#NumericFocus=1, IterationLimit=10,  #solver = 'ECOS_BB')#verbose = True, , NumericFocus=3 warmstart = True, 


    toc = time.time()-tic
    print('optimal cost: '+ str(result))
    print('time step: ' + str(timestep))
    #if result != float('inf'):
    #    sort_solution(prob, var_name_list,T, forecast, timestep)#print(prob._solution)
    print('problem solved in '+str(toc)+'seconds')

    # find error in voltage convergence
    #voltage_error = [x_m[i,t].value[0] - x_n[i,t] for i,t in product(range(n_nodes), range(T))]
    voltage_error = np.zeros((n_e_nodes, T))
    pid_error = np.zeros((n_e_nodes, T))

    for m in range(n_e_nodes):
        if not m==6:
            for t in range(T):
                # calculate error
                voltage_error[m,t] = x_m[m,t].value[0] - x_n[m,t]
                x_n[m,t] = x_n[m,t] + voltage_error[m,t]*0.5#*(1-0.01/3)
            

    #if values have converged, save values
    if np.all(abs(voltage_error)<= 0.005) or v_iters>=19:
        pid_error = np.zeros((n_e_nodes, T))
        if v_iters >=19:
            print('no convergence after 20 iterations, start clean for next timestep')
            x_n = np.ones((n_e_nodes,T))
        else:
            print('voltage value convergence after '+ str(v_iters) + ' iterations')
        v_iters = 0
        print('parsing and saving solution')
        tic = time.time()
        ######## SORT PROBLEM SOLUTION
        #create a csv of the results
        # make the first row, the name list
        filename = 'command_output_wsu.csv'
        field_names = []
        if timestep==0:
            open_method = 'w'        
        else:
            open_method = 'a'
        
        with open(filename, open_method) as logfile:
            values = {}
            for i in range(len(var_name_list)):
                var_name = var_name_list[i]
                split_name = var_name.split('_')
                var_name = var_name.split(split_name[-2])[0][:-1]
                j = int(split_name[-2])
                t = int(split_name[-1])
                field_name = var_name+'_'+str(j)
                # get numeric value
                var_val = eval(var_name)[j,t]
                if var_val.attributes['boolean']:
                    var_val = var_val.value
                elif var_val.value == None:
                    var_val = 0
                else:
                    var_val = var_val.value[0]
                # add to entry
                if field_name in values:
                    values[field_name].append(var_val)
                else:
                    field_names.append(field_name)
                    values[field_name] = [var_val]
            field_names.append('solar')
            field_names.append('ep_demand')
            field_names.append('eq_demand')
            field_names.append('h_demand')
            field_names.append('c_demand')
            values['solar'] = sum(forecast.renew)
            values['ep_demand'] = sum(forecast.demand.ep)
            values['eq_demand'] = sum(forecast.demand.eq)
            values['h_demand'] = sum(forecast.demand.h)
            values['c_demand'] = sum(forecast.demand.c)
            
            logger = csv.DictWriter(logfile, fieldnames = field_names, lineterminator = '\n')
            if timestep==0:
                logger.writeheader()
            # else:
            #     logger = csv.writer(logfile)
            for t in range(T):
                values_by_row = {}
                for key, value in values.items():
                    values_by_row[key] = value[t]
                #if timestep==0:
                logger.writerow(values_by_row)
                #else:
        toc = time.time()-tic
        print('time for parsing '+ str(toc))

    del prob
    return v_iters, x_n, pid_error





for t in range(timesteps):
    v_iters = 1
    pid_error = np.zeros((n_e_nodes, T))
    # allow up to ten iterations before just acceptig the last iteration as close enough
    while v_iters>0 and v_iters<20:
        constraints = []
        var_name_list = []
        # if you are on a subsequent timestep, try the voltages from the last timestep before starting over
        # at the maximum voltage deviation. This helps reduce iterations, but may cause an overshoot in later timesteps
        if v_iters == 1 and t>0:
            v_iters, x_n, pid_error = run_horizon(t, v_iters, x_n, pid_error)

        else: 
            v_iters, x_n, pid_error = run_horizon(t, v_iters, x_n, pid_error)
        # increment the iteration
        if v_iters>0:
            v_iters = v_iters+1
    
    if v_iters==20:
        error_string = 'Error: voltage iteration limit exceeded, continuing with best approximation for timestep '+str(t)
        print(error_string)

    # update x_n for next timestep
    for n in range(n_e_nodes):
        x_n_end = x_n[n][0]
        x_n[n][:-1] = x_n[n][1:]
        x_n[n][-1] = x_n_end
        #x_n[n][-1] = 1.0

    # update dates
    start_date = start_date + datetime.timedelta(hours=1)
    date_range = [start_date+datetime.timedelta(hours=i) for i in range(T)]
    #update initial conditions
    # read previous horizon's entry
    n_row = t*T+1
    # get the header indexing on the first iteration
    if t == 0:
        with open('command_output_wsu.csv', 'r') as filename:
            logger = csv.reader(filename)
            headers = next(logger, None)
        t_init_i = headers.index('turbine_xp_0')
    




        