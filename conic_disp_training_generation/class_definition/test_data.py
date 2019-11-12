'''
Defines the TestData class and supporting classes.
TestData contains information necessary for running EAGERS in simulation
mode (i.e. when not connected to a real building).
'''

from class_definition.specifiable import Specifiable


class TestData(Specifiable):
    '''Test Data class.

    ATTRIBUTES:
    timestamp
    demand
    weather
    hist_prof
    real_time_data
    '''

    def __init__(self, **kwargs):
        self.timestamp = []
        self.demand = Demand()
        self.weather = Weather()
        self.hist_prof = []
        self.real_time_data = []

        self.set_attrs(**kwargs)


class Demand(Specifiable):
    '''Demand class.

    ATTRIBUTES:
    e
    h
    c
    '''

    def __init__(self, **kwargs):
        self.e = []
        self.h = []
        self.c = []

        self.set_attrs(**kwargs)


class Weather(Specifiable):
    '''Weather class.

    ATTRIBUTES:
    t_db
    irrad_dire_norm
    '''

    def __init__(self, **kwargs):
        self.t_db = []
        self.irrad_dire_norm = []
        
        self.set_attrs(**kwargs)
