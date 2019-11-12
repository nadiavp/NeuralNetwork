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

    avista_115 = setup_elec_utility(name='TUR115')
    gt1 = setup_mgt()
    gt2 = setup_mgt(n_gt=2)
    carrier_chiller1 = setup_chiller('Carrier Chiller1')
    york_chiller1 = setup_chiller('York Chiller1')
    york_chiller3 = setup_chiller('York Chiller3')
    carrier_chiller7 = setup_chiller('Carrier Chiller7')
    carrier_chiller8 = setup_chiller('Carrier Chiller8')
    carrier_chiller2 = setup_chiller('Carrier Chiller2')
    carrier_chiller3 = setup_chiller('Carrier Chiller3')
    carrier_chiller4 = setup_chiller('Carrier Chiller4')
    trane_chiller = setup_chiller('Trane Chiller')
    boiler1 = setup_boiler(1)
    boiler2 = setup_boiler(2)
    boiler3 = setup_boiler(3)
    boiler4 = setup_boiler(4)
    boiler5 = setup_boiler(5)
    gt3 = setup_mgt(n_gt=3)
    gt4 = setup_mgt(n_gt=4)
    #gen_3 = setup_diesel_gen()
    cold_water_tank = setup_cold_thermal_storage()
    rooftop_pv = setup_solar()
    ground_pv = setup_solar(rooftop=False)
    gas_utility = setup_gas_utility()
    diesel_utility = setup_diesel_utility()
    avista_125 = setup_elec_utility(name='SPU125')
    avista_122 = setup_elec_utility(name='SPU122')
    avista_124 = setup_elec_utility(name='SPU124')
    avista_111 = setup_elec_utility(name='TUR111')
    avista_131 = setup_elec_utility(name='TVW131')
    avista_117 = setup_elec_utility(name='TUR117')

    components = [avista_115, gas_utility, gt1, carrier_chiller1, cold_water_tank, 
        boiler1, york_chiller1, york_chiller3, carrier_chiller7, carrier_chiller8, carrier_chiller2,
        carrier_chiller3, carrier_chiller4, trane_chiller, gt2, gt3, gt4, boiler2, boiler3, boiler4,
        boiler5, rooftop_pv, ground_pv, diesel_utility, avista_125, avista_122, avista_124, avista_111, avista_131, avista_117]

    optimoptions = {'interval':365, 'horizon':24, 'resoluion':1, 'excess_heat':True, 'mixed_integer':False, 'excess_cool':True}
    campus_network = load_network(components)#load_network_one_node(components)#
    plant = Plant({'name': 'WSU_campus_DC', 'generator':components, 'optimoptions':optimoptions, 'network': campus_network})

    gen_file_name = os.path.join('library', 'wsu_campus.pickle')
    with open(gen_file_name, 'wb') as write_file:
        pickle.dump(plant, write_file, pickle.HIGHEST_PROTOCOL)





#establish utilities
def setup_elec_utility(name='Elec_Utility'):
    rt_summer = [[0]*24]*7#np.ones((7,24))
    i_day = 0
    for day in rt_summer:
        rt_summer[i_day][8:20] = [2]*12
        i_day = i_day+1
    r_summer = np.array([
        [0.03, 0],
        [0.0551, 0],
        [0.07, 0]
    ])
    rt_winter = rt_summer
    r_winter = np.array([
        [0.04, 0],
        [0.0551, 0],
        [0.06, 0]
    ])
    util = Utility(
        name = name, 
        sum_rate_table = rt_summer,
        win_rate_table = rt_winter,
        sum_rates = r_summer,
        win_rates = r_winter,
        sum_start_month = 6,
        sum_start_day = 1,
        win_start_month = 10,
        win_start_day = 1,
    )
    return util

def setup_gas_utility():
    ts = date_range((2009,1,1), (2014,1,1))
    util = Utility(
        name = 'Gas Utility',
        source = 'ng',
        size = 0,
        timestamp = ts,
        rate = np.linspace(5.6173, 5.6173, num = len(ts))
    )
    return util

def setup_diesel_utility():
    ts = date_range((2009,1,1), (2014,1,1))
    util = Utility(
        name = 'Diesel Utility',
        source = 'diesel',
        size = 0,
        timestamp = ts,
        rate = np.linspace(24, 24, num=len(ts))
    )
    return util

#gas turbines are similar
def setup_mgt(n_gt=1):
    cap1 = np.array([
        0,
        0.00110499999998137,
        0.00361777777760290,
        0.385651944444515,
        0.388592222222127,
        0.431189166666474,
        0.484720833333675,
        0.491660000000149,
        0.505131388888694,
        0.586546944444534,
        0.621058055555681,
        1.05868694444420,
        1.06167222222197,
        1.07020861111116,
        1.07115888888901
    ])
    cap2 = np.array([
        0,
        0.000956666666781530,
        0.00366805555578321,
        0.388506666666362,
        0.396046111111296,
        0.438080833333079,
        0.493104722222313,
        0.497185555555392,
        0.515001388888806,
        0.590879722222220,
        0.629853611110942,
        1.07497166666691,
        1.07684916666662,
        1.08608277777792,
        1.08781305555580
    ])
    elec1 = np.array([
        0,
        0.686800696496086,
        0.510127443690416,
        0.309220510095308,
        0.312715841910826,
        0.306017287041821,
        0.362823346550335,
        0.353123964797020,
        0.315108043596381,
        0.368046107155067,
        0.358760399187536,
        0.344788257890904,
        0.343248719213812,
        0.343656819850057,
        0.344227297586808
    ])
    elec2 = np.array([
        0,
        0.459853195951741,
        0.428405275836534,
        0.316385631927067,
        0.313322663749180,
        0.312953937427671,
        0.362498468567169,
        0.364462640163305,
        0.313690064224268,
        0.365898726735581,
        0.358972505189864,
        0.347214900723712,
        0.353977200593419,
        0.349119253281163,
        0.348170539023118
    ])
    heat = np.ones((15))*0.5
    if n_gt ==1:
        out = Output(capacity = cap1, electricity = elec1, heat = heat)
    else:
        out = Output(capacity = cap2, electricity = elec2, heat = heat)
        
    ss = StateSpace(
        a = np.matrix('0,1; -0.3085, -225.8605'),
        b = np.matrix('0,0.3085'),
        c = np.matrix('1,0'),
        d = np.matrix('0')
        )
    startup1 = Startup(
        time = np.array([0,0.1]),
        electricity = np.array([0, 0.129387]),
        input = np.array([0,0.548259]),
        heat = np.array([0,0.271938])
    )
    startup2 = Startup(
        time = np.array([0,10]),
        electricity = np.array([0, 12.9387]),
        input = np.array([0,54.8259]),
        heat = np.array([0,27.1938])
    )
    shutdown= Shutdown( time = np.array([0,1000]),
        electricity = np.array([1.2939e3, 0]),
        input = np.array([5.4826e3,0]),
        heat = np.array([2.7194e3,0])
    )
    #comm = Comm(on_off = 51700, set = 51100),
    #measure = Measure(on_off = 32300, input = 31200, electric=31100, thermal =31300),
    mgt = MicroTurbine(
        name = 'GT1',
        output = out,
        size = 5000,
        startup = startup1,
        shutdown = shutdown,
        start_cost = 323.4671,
        restart_time = 15,
        ramp_rate = 1.3344e3
    )
    if n_gt ==2:
        mgt.name = 'GWSPE'
        mgt.size = 43750
        mgt.startup = startup2
    elif n_gt == 3:
        mgt.name = 'GWSPB'
        mgt.size = 2750
    elif n_gt == 4:
        mgt.name = 'GWSPA'
        mgt.size = 2750
    return mgt

#first chiller
def setup_chiller(name_str):
    wb = xlrd.open_workbook('instance\chiller_capacity_output.xlsx')
    sheet = wb.sheet_by_index(0)
    cap = []
    cooling = []
    if name_str=='Carrier Chiller1':
        chiller_size = 7.279884675000000e+03
        dx_dt = 4.8533e3
        ss = StateSpace(a = np.matrix('0,1; -1.9997e-7, -0.1278'), b = np.matrix('0; 1.9997e-7'),
            c = np.matrix('1,0'), d = np.matrix('0'))
        stu = Startup(time = np.array([0, 0.1000]), cooling = np.array([0, 36.3994]), input = np.array([0, 0.0050]))
        shd = Shutdown(time = np.array([0, 1000]), cooling = np.array([3.6399e3, 0]), input = np.array([0.4999, 0]))
        for i in range(2,482):
            cap.append(sheet.cell_value(i,0))
            cooling.append(sheet.cell_value(i,1))
    elif name_str[:-1]=='York Chiller':
        chiller_size = 5.268245045000001e+03
        dx_dt = 3.5122e3
        ss = StateSpace(a = np.matrix('0, 1; -3.8187e-7, -0.1767'), b = np.matrix('0; 3.8187e-7'),
            c = np.matrix('1,0'), d = np.matrix('0'))
        stu = Startup(time = np.array([0, 10]), cooling = np.array([0, 26.3764]), input = np.array([0, 0.005]))
        shd = Shutdown(time = np.array([0, 1000]), cooling = np.array([2.6376e3, 0]), input = np.array([0.4998, 0]))
        for i in range(2, 129):
            cap.append(sheet.cell_value(i,2))
            cooling.append(sheet.cell_value(i,3))
    elif name_str=='Carrier Chiller2' or name_str=='Carrier Chiller3':
        chiller_size = 4.853256450000000e+03
        dx_dt = 3.2355e3
        ss = StateSpace(a = np.matrix('0,1; -4.4997e-7, -0.1918'), b = np.matrix('0; 4.4997e-7'),
            c = np.matrix('1,0'), d = np.matrix('0'))
        stu = Startup(time = np.array([0, 10]), cooling = np.array([0, 24.2663]), input = np.array([0, 0.005]))
        shd = Shutdown(time = np.array([0, 1000]), cooling = np.array([2.4266e3, 0]), input = np.array([0.4997, 0]))
        for i in range(2, 423):
            cap.append(sheet.cell_value(i,10))
            cooling.append(sheet.cell_value(i,11))
    elif name_str=='Carrier Chiller7' or name_str=='Carrier Chiller8':
        chiller_size = 5.275278750000000e+03
        dx_dt = 3.5169e3
        ss = StateSpace(a = np.matrix('0,1; -3.8085e-7, -0.1764'), b = np.matrix('0; 3.8085e-7'),
            c = np.matrix('1,0'), d = np.matrix('0'))
        stu = Startup(time = np.array([0,10]), cooling = np.array([0, 26.3764]), input = np.array([0, 0.005]))
        shd = Shutdown(time = np.array([0, 1000]), cooling = np.array([2.6376e3, 0]), input = np.array([0.4998, 0]))
        for i in range(2, 663):
            cap.append(sheet.cell_value(i,6))
            cooling.append(sheet.cell_value(i,7))
    elif name_str=='Carrier Chiller4':
        chiller_size = 1.758426250000000e+03
        dx_dt = 1.1723e3
        ss = StateSpace(a= np.matrix('0, 1; -3.4289e-6, -0.5294'), b = np.matrix('0; 3.4289e-6'),
            c = np.matrix('1,0'), d = np.matrix('0'))
        stu = Startup(time = np.array([0,10]), cooling = np.array([0, 8.7921]), input = np.array([0, 0.005]))
        shd = Shutdown(time = np.array([0, 1000]), cooling = np.array([879.2131, 0]), input = np.array([0.4995,0]))
        for i in range(2,7):
            cap.append(sheet.cell_value(i,14))
            cooling.append(sheet.cell_value(i,15))
    else: #if it is the trane chiller
        chiller_size = 1.415462794200000e+03
        dx_dt = 943.6419
        ss = StateSpace(a = np.matrix('0, 1; -5.2926e-6, -0.6577'), b = np.matrix('0; 5.2926e-6'),
            c = np.matrix('1,0'), d = np.matrix('0'))
        stu = Startup(time = np.array([0, 10]), cooling = np.array([0, 7.0773]), input = np.array([0, 0.005]))
        shd = Shutdown(time = np.array([0, 1000]), cooling = np.array([707.7314, 0]), input = np.array([0.4996, 0]))
        for i in range(2, 168):
            cap.append(sheet.cell_value(i,16))
            cooling.append(sheet.cell_value(i,17))
    op = Output(capacity = np.array(cap), cooling = np.array(cooling))
    chiller = ElectricChiller(
        name = name_str,
        size = chiller_size,
        output = op,
        state_space = ss,
        startup = stu,
        shutdown = shd,
        start_cost = 0,
        ramp_rate = dx_dt,
        source = 'electricity'
    )
    return chiller

#all boilers look the same
def setup_boiler(boiler_num=1):
    name_str = 'Boiler'+str(boiler_num)
    op = Output(capacity = np.array([
        0,
        0.250000000000000,
        0.500000000000000,
        0.700000000000000,
        0.760000000000000,
        0.800000000000000,
        0.840000000000000,
        0.880000000000000,
        0.920000000000000,
        0.960000000000000,
        1
    ]), 
    heat = np.ones(11))
    ss = StateSpace(a = np.matrix('0,1; -8.7249e-9, -1.8681e-4'), b = np.matrix('0; 8.7249e-9'), c = np.matrix('1,0'), d=np.matrix('0'))
    stu = Startup(time = np.array([0,10]), heat = np.array([0,2]), input = np.array([0,2]))
    shd = Shutdown(time = np.array([0,1000]), heat = np.array([200,0]), input = np.array([200,0]))

    boiler = Heater(
        name = name_str,
        size = 20000,
        output = op,
        state_space = ss,
        startup = stu,
        shutdown = shd,
        start_cost = 0,
        ramp_rate = 1.3333e3
    )
    return boiler

#diesel generator is different from the GT's
def setup_diesel_gen():
    op = Output(capacity = np.array([0,
        0.0440436111111194,
        0.0959544444443891,
        0.274149999999907,
        1.00805555555550,
        1.13232999999996]), 
        electricity = np.array([
        0,
        0.0386165028943384,
        0.0640837968864363,
        0.276878400371175,
        0.429227220146803,
        0.567247108410006
        ]))
    ss = StateSpace(a = np.matrix('0,1; -1.924e-6, -0.0028'), b = np.matrix('0; 1.9240e-6'),
        c = np.matrix('1,0'), d = np.matrix('0'))
    stu = Startup(time = np.array([0, 1000]), electricity = np.array([0, 333.3333]), input = np.array([0, 1.2979e3]))
    shd = Shutdown(time = np.array([0, 1000]), electricity = np.array([333.3333,0]), input = np.array([1.2979e3,0]))
    start_cost = 66.6667
    dx_dt = 666.6692
    gen = ElectricGenerator(name = 'Gen3', source = 'diesel', size = 1000,
        output = op, state_space = ss, startup = stu, shutdown = shd, 
        start_cost = start_cost, ramp_rate = dx_dt)
    return gen


#both solar arrays look similar
def setup_solar(rooftop=True):
    op = Output(capacity = np.array([
        0,
        0.100000000000000,
        0.200000000000000,
        0.300000000000000,
        0.400000000000000,
        0.500000000000000,
        0.600000000000000,
        0.700000000000000,
        0.800000000000000,
        0.900000000000000,
        1
    ]), electricity = np.ones(11))
    array = Solar(
        name = 'Rooftop PV',
        output = op,
        size = 600,
        eff = .1740,
        size_m2 = 194.04,
        tilt = 20,
        azimuth = 180,
        gen_frac = 0.8719,
        dem_frac = 0.9247,
        us_state = 'Washington',
        data =np.matrix(['.95, .8, 1.05; .92, .88, .98;\
         .98, .97, .995; .995, .99, .997; .98, .97, .99; .99, .98, .993; .95, .3, .995;\
         .98, 0, .995; 1, 0, 1; 1, .95, 1; 1, .7, 1'])
    )
    if not rooftop:
        array.name = 'Ground PV'
        array.size = 900
        array.size_2m = 352.80,
        array.tilt = 65
    return array

#only one cold thermal storage tank
def setup_cold_thermal_storage():
    cold_storage = ThermalStorage(name = 'Cold Water Tank', 
        source = 'cooling',
        output = [],
        size = 2000000,
        size_liter = 7570820,
        t_cold = 4,
        t_hot = 13,
        fill_rate = 3000000,
        fill_rate_perc = 20,
        disch_rate = 3000000,
        disch_rate_perc = 20,
        self_discharge = 0.0017,
        disch_eff = 0.99,
        charge_eff = 0.99,
        ramp_rate = 400000,
        buffer = 20,
        peak_disch = 1000000*.95,
        max_dod = 1000000*.2
    )
    return cold_storage

def date_range(start, end):
    start_dt = datetime(*start)
    end_dt = datetime(*end)
    return [start_dt + timedelta(1)*i for i in range((end_dt-start_dt).days + 1)]

#load all the network components into their respective nodes
def load_network_one_node(gens):
    names = 'central node'
    equipment = [gens]
    e_connections = []
    h_connections = []
    c_connections = []
    longitude = [-117.1698]
    latitude = [46.7287]
    timezone = -3
    location = Location()
    location.longitude = longitude[0]
    location.latitude = latitude[0]
    location.timezone = timezone
    e_trans_eff = []
    h_trans_eff = []
    c_trans_eff = []
    e_load = [[0]]
    h_load = [[0]]
    c_load = [[0]]
    network = []
    e_demand = NetworkDemand({'connections': e_connections, 'trans_eff': e_trans_eff, 'trans_limit':[], 'load': e_load[0]})
    h_demand = NetworkDemand({'connections':h_connections, 'trans_eff':h_trans_eff, 'trans_limit':[], 'load':h_load[0]})
    c_demand = NetworkDemand({'connections':c_connections, 'trans_eff':c_trans_eff, 'trans_limit':[], 'load':c_load[0]})
    node = Network(gens=True, info_dct={'equipment':equipment[0], 'name':names, 'electrical':e_demand, 'district_heat':h_demand, 'district_cooling':c_demand, 'location': location})
    network.append(node)
    return network


def load_network(gens):
    names = ['TUR115', 'SPU125', 'SPU122', 'SPU124', 'TUR111', 'TUR131', 'TUR117']
    equipment = [[gens[0], gens[1], gens[2], gens[18]],\
         [gens[6], gens[7], gens[3], gens[24]],\
         [gens[1], gens[5], gens[14], gens[15], gens[25], gens[20]],\
         [gens[1], gens[12], gens[13], gens[16], gens[17], gens[8], gens[9], gens[26]],\
         [gens[27]],\
         [gens[1], gens[10],gens[11], gens[4], gens[19], gens[28]],\
         [gens[29], gens[21], gens[22]]]
    e_connections = [['SPU125'], ['TUR115'],['SPU124'], ['SPU122'], [],[],[]]
    h_connections = [['TUR131', 'TUR111'],[],['TUR111', 'SPU124'], ['SPU122', 'TUR131'],['TUR115', 'SPU122'],['SPU124', 'TUR115'], []]
    c_connections = [[],['TUR111', 'SPU122'],['SPU125', 'TUR131'], ['TUR131','TUR111'],['SPU124', 'SPU125'],['SPU122', 'SPU124'],[]]
    longitude = [-117.1698, -117.1527, -117.1558, -117.16, -117.1616, -117.1529, -117.1515, -117.1710, -117.1626, -177, -177]
    latitude = [46.7287, 46.7326, 46.7320, 46.7317, 46.7298, 46.7295, 46.7287, 46.7296, 46.7444, 46.7, 46.7]
    timezone = -6
    eff = []
    e_trans_eff = [eff]*len(names)#[[.999],[.95],[.999, .95, .97],[.999, .95, .97],[.999], [.95, .95, .95, .95, .95, .95], [.95, .97, .97, .95], [.999, .95], [.9], [.9], [.9]]
    h_trans_eff = [eff]*len(names)#[[], [], [.91,.9], [], [.9,.9], [.91,.98], [.9, .9, .98], [.9], [], [], [.9]]
    c_trans_eff = [eff]*len(names)#[[.85],[.95], [.95,.98], [.98,.93], [.85, .93], [], [], [], [], [9], []]
    e_load = [2, 2, 0, 1, 3, 2, 3]
    h_load = [None, None, 0, 1, 2, 3, None]
    c_load = [None, None, 0, 1, 2, 3, None]
    network = []
    for i in range(len(names)):
        location = Location()
        location.longitude = longitude[i]
        location.latitude = latitude[i]
        location.timezone = timezone
        e_demand = NetworkDemand({'connections': e_connections[i], 'trans_eff': e_trans_eff[i], 'trans_limit': np.ones(len(e_trans_eff[i]))*np.float('inf'), 'load': e_load[i]})
        h_demand = NetworkDemand({'connections': h_connections[i], 'trans_eff': h_trans_eff[i], 'trans_limit': np.ones(len(h_trans_eff[i]))*np.float('inf'), 'load': h_load[i]})
        c_demand = NetworkDemand({'connections': c_connections[i], 'trans_eff': c_trans_eff[i], 'trans_limit': np.ones(len(c_trans_eff[i]))*np.float('inf'), 'load': c_load[i]})
        node = Network(gens = True, info_dct={'equipment':equipment[i], 'name': names[i], 'electrical': e_demand, 'district_heat': h_demand, 'district_cooling': c_demand, 'location': location})
        network.append(node)
    return network


pickle_wsu()