import os
import pickle
import copy
import numpy as np
import xlrd
from datetime import datetime, timedelta

from class_definition.component import (Utility, MicroTurbine, ElectricGenerator, Heater, ThermalStorage, ElectricChiller, Solar)
from class_definition.generator_struct import (Output, StateSpace, Startup, Shutdown, Comm, Measure)
from class_definition.plant_struct import (Optimoptions, Network, Location, NetworkDemand, Plant)

def pickle_wsu():
    #create all components in network for the wsu campus

    slack_gt = setup_electric_gen(name='slack_bus_1')
    GT1 = setup_electric_gen(name='GT1')
    GT2 = setup_electric_gen(name='GT2')
    gas_util = setup_gas_utility()

    components = [slack_gt, GT1, GT2, gas_util]

    optimoptions = {'interval':1, 'horizon':1, 'resoluion':1, 'excess_heat':True, 'mixed_integer':False, 'excess_cool':True}
    campus_network = load_network(components)#load_network_one_node(components)#
    plant = Plant({'name': 'demo_7_9', 'generator':components, 'optimoptions':optimoptions, 'network': campus_network})

    gen_file_name = os.path.join('library', 'demo_7_9.pickle')
    with open(gen_file_name, 'wb') as write_file:
        pickle.dump(plant, write_file, pickle.HIGHEST_PROTOCOL)



#establish non-CHP generators
def setup_electric_gen(name='GT0'):
    p_base = 100
    if name == 'slack_bus_1':
        size = 85
        fundata =  {"bp": 0.008, "fp": 7.0, "cp": 200}
        lb = 0
    elif name == 'GT1':
        size = 80
        fundata =  {"bp": 0.009, "fp": 6.3, "cp": 180}
        lb = 0
    elif name == 'GT2':
        size = 70
        fundata =  {"bp": 0.007, "fp": 6.8, "cp": 140}
        lb = 0
    gt = ElectricGenerator(name = name,
        size = size,
        ub = size/p_base,
        lb = lb/p_base,
        fundata = fundata,
        start_cost = 323.4671,
        restart_time = 15,
        ramp_rate = 1.3344e3)
    return gt



def setup_gas_utility():
    ts = date_range((2009,1,1), (2014,1,1))
    util = Utility(
        name = 'Gas Utility',
        source = 'ng',
        size = 0,
        timestamp = ts,
        rate = np.linspace(293.07, 293.07, num = len(ts))
    )
    return util




def date_range(start, end):
    start_dt = datetime(*start)
    end_dt = datetime(*end)
    return [start_dt + timedelta(1)*i for i in range((end_dt-start_dt).days + 1)]

def load_network(gens):
    names = ['slack', 'bus_2', 'bus_3', 'bus_4', 'bus_5']
    equipment = [[gens[0], gens[3]], [gens[1], gens[3]], [gens[2], gens[3]],[],[]]
    e_connections = [['bus_2', 'bus_3'], ['slack', 'bus_3', 'bus_4', 'bus_5'], ['slack', 'bus_2', 'bus_4'], ['bus_2', 'bus_3', 'bus_5'],['bus_2', 'bus_4']]
    longitude = [-117.1698, -117.1527, -117.1558, -117.16, -117.1616]
    latitude = [46.7287, 46.7326, 46.7320, 46.7317, 46.729]
    timezone = -6
    eff = []
    e_trans_eff = [eff]*len(names)
    e_load = [[0,0],[20,10],[20,15],[50,30],[60,40]]
    network = []
    for i in range(len(names)):
        location = Location()
        location.longitude = longitude[i]
        location.latitude = latitude[i]
        location.timezone = timezone
        e_demand = NetworkDemand({'connections': e_connections[i], 'trans_eff': e_trans_eff[i], 'trans_limit': np.ones(len(e_trans_eff[i]))*np.float('inf'), 'load': e_load[i]})
        h_demand = NetworkDemand({'connections': [], 'trans_eff': [], 'trans_limit': [], 'load': None})
        c_demand = NetworkDemand({'connections': [], 'trans_eff': [], 'trans_limit': [], 'load': None})
        node = Network(gens = True, info_dct={'equipment':equipment[i], 'name': names[i], 'electrical': e_demand, 'district_heat': h_demand, 'district_cooling': c_demand, 'location': location})
        network.append(node)
    return network


pickle_wsu()