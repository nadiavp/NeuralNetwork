'''
Defines classes used in high level operation of EAGERS.
Optimoptions: Options used to influence the behavior of the optimizer.
Network: Holds energy network information.
'''

class Plant:
    def __init__(self, info_dct=None):
        if not info_dct == None:
            for key in info_dct:
                setattr(self, key, info_dct[key])
        else:
            self.name = 'plant1'
            self.generator = []
            self.optimoptions = Optimoptions()
            self.network = Network(self.generator)
            self.subnet = SubNet(self.generator)


class Optimoptions:
    '''Optimization options class.

    ATTRIBUTES:
    interval
    horizon
    resolution
    t_opt
    t_mpc
    scaletime
    t_spacing
    excess_heat
    threshold_steps
    method
    mixed_integer
    spin_reserve
    spin_reserve_perc
    forecast
    solver
    mode
    excess_cool
    end_soc
    '''

    def __init__(self, info_dct=None):
        if not info_dct == None and isinstance(info_dct, dict):
            # Load in this dictionary's information.
            for key in info_dct:
                setattr(self, key, info_dct[key])
        else:
            # Load defaults.
            self.interval = 31
            self.horizon = 24
            self.resolution = 1
            self.t_opt = 3600
            self.t_mpc = 600
            self.scaletime = 1
            self.t_spacing = 'constant'
            self.excess_heat = True
            self.threshold_steps = 6
            self.method = 'dispatch'
            self.mixed_integer = True
            self.spin_reserve = False
            self.spin_reserve_perc = 0
            self.forecast = 'perfect'
            self.solver = 'quadprog'
            self.mode = 'virtual'
            self.excess_cool = False
            self.end_soc = 'flexible'


class Network:
    '''Network class.

    ATTRIBUTES:
    name
    equipment
    (electrical)
    (district_heat)
    (district_cool)
    direct_current
    location
    '''

    def __init__(self, gens, info_dct=None):
        if info_dct is not None:
            # Load in this dictionary's information.
            for key in info_dct:
                setattr(self, key, info_dct[key])
        else:
            # Load defaults.
            self.name = 'node1'
            self.equipment = []
            # Load the names of the generators and check for demand
            # types.
            demand_type = []
            for g in gens:
                self.equipment.append(g.name)
                if hasattr(g.output, 'electricity'):
                    demand_type.append('electrical')
                if hasattr(g.output, 'heat'):
                    demand_type.append('district_heat')
                if hasattr(g.output, 'cooling'):
                    demand_type.append('district_cool')
            for dem_type in demand_type:
                setattr(self, dem_type, NetworkDemand())
                
            # Load the location coordinates and direct current demands.
            self.location = Location()
            self.direct_current = NetworkDemand()
            self.direct_current.load = []


class NetworkDemand:
    def __init__(self, info_dct = None):
        if not info_dct == None:
            for key in info_dct:
                setattr(self, key, info_dct[key])
        else:
            self.connections = []
            self.trans_eff = 1
            self.trans_limit = float('inf')
            self.load = True

class SubNet:
    def __init__(self, gens):
        '''Load the names of the generators, and check for demand types.'''
        # Use of tuple ensures a certain order, as opposed to dictionary keys.
        demand_types = ('electricity', 'heat', 'cooling', 'ac_to_dc_eff')
        attr_name = dict(
            electricity = 'electrical',
            heat = 'district_heat',
            cooling = 'district_cool',
            ac_to_dc_eff = 'direct_current',
        )
        for d in demand_types:
            for g in gens:
                if hasattr(g.output, d):
                    setattr(self, attr_name[d], SubNetProperties())
                    break
                if d == 'ac_to_dc_eff' and hasattr(g, d):
                    setattr(self, attr_name[d], SubNetProperties())
                    break


class SubNetProperties:
    def __init__(self):      
        #load the location coordinates and direct current demands
        self.name = 'node1'
        self.nodes = []
        self.location = Location()
        self.abbreviation = []
        self.line_names = []
        self.line_number = []
        self.line_limit = []
        self.line_eff = []
        self.equipment = []


class Location:
    def __init__(self):
        #default to colorado
        self.longitude = -105
        self.latitude = 40
        self.timezone = -6
