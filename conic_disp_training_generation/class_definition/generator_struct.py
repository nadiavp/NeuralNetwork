'''
Defines classes used to hold generator information.
'''

import numpy as np


class Output:
    def __init__(self, fields=[], **kwargs):
        for f in fields:
            setattr(self, f, [])
        for key in kwargs:
            setattr(self, key, kwargs[key])


class StateSpace:
    def __init__(self, **kwargs):
        for state in ['a', 'b', 'c', 'd']:
            setattr(self, state, [])
        for key in kwargs:
            setattr(self, key, kwargs[key])


class Startup:
    def __init__(self, shutdown=[], **kwargs):
        if isinstance(shutdown, Shutdown):
            for key in vars(shutdown).keys():
                if key == 'time':
                    setattr(self, key, getattr(shutdown, key))
                else:
                    setattr(self, key, np.flip(getattr(shutdown, key), 0))
        for key in kwargs:
            setattr(self, key, kwargs[key])


class Shutdown:
    def __init__(self, startup=[], **kwargs):
        if isinstance(startup, Startup):
            for key in vars(startup).keys():
                if key == 'time':
                    setattr(self, key, getattr(startup, key))
                else:
                    setattr(self, key, np.flip(getattr(startup, key), 0))
        for key in kwargs:
            setattr(self, key, kwargs[key])


class Comm:
    def __init__(self, on_off=None, set_pt=None):
        self.on_off = on_off
        self.set = set_pt


class Measure:
    def __init__(self, on_off=None, inpt=None, electric=None, thermal=None):
        self.on_off = on_off
        self.input = inpt
        self.electric = electric
        self.thermal = thermal
