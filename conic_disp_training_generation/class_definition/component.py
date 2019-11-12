"""
Defines component classes. Each component is a child of the top most
parent class, Component. Heirarchy is as follows:

Component
|__ Generator
|   |__ Chiller
|   |   |__ ElectricChiller
|   |   |__ AbsorptionChiller
|   |
|   |__ Heater
|   |   |__ AirHeater
|   |   |__ WaterHeater
|   |
|   |__ CoolingTower
|   |
|   |__ HydrogenGenerator
|   |   |__ Electrolyzer
|   |
|   |__ CombinedHeatPower
|   |   |__ InternalCombustionEngine
|   |   |__ FuelCell
|   |   |   |__ ReversibleFuelCell
|   |   |__ MicroTurbine
|   |
|   |__ ElectricGenerator
|   |
|   |__ Renewable
|       |__ Solar
|       |__ Wind
|       |__ Hydro
|
|__ Storage
|   |__ ElectricStorage
|   |__ ThermalStorage
|   |__ HydrogenStorage
|   |__ HydroStorage
|
|__ Utility
|   |__ DistrictHeat
|   |__ DistrictCool
|
|__ ACDCConverter
"""

from class_definition.specifiable import Specifiable
from class_definition.generator_struct import (Output, StateSpace, Startup,
    Shutdown, Comm, Measure)


class Component(Specifiable):
    '''Parent class for all components.
    
    ATTRIBUTES:
    name
    enabled
    [output]
    '''
    def __init__(self, output_fields=[], **kwargs):
        self.set_attrs(
            name = 'component',
            enabled = True,
            output = self.output_from_fields(output_fields),
        )
        self.set_attrs(**kwargs)

    def output_from_fields(self, output_fields):
        return Output(fields=output_fields)

class Generator(Component):
    '''Parent class for all generators.

    ATTRIBUTES:
    source
    size
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'generator',
            source = 'source',
            size = 0,  # [kW]
        )
        self.set_attrs(**kwargs)

class Chiller(Generator):
    '''Chiller class.

    ATTRIBUTES:
    state_space
    startup
    shutdown
    start_cost
    ramp_rate
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'chiller',
            state_space = StateSpace(),
            startup = Startup(),
            shutdown = Shutdown(),
            start_cost = 0,  # [$]
            ramp_rate = 0,  # [kW/hr]
        )
        self.set_attrs(**kwargs)

class ElectricChiller(Chiller):
    '''Electric Chiller class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'electric chiller',
        )
        self.set_attrs(**kwargs)

class AbsorptionChiller(Chiller):
    '''Absorption Chiller class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'absorption chiller',
        )
        self.set_attrs(**kwargs)

class Heater(Generator):
    '''Heater class.

    ATTRIBUTES:
    state_space
    startup
    shutdown
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'heater',
            source = 'ng',
            state_space = StateSpace(),
            startup = Startup(),
            shutdown = Shutdown(),
        )
        self.set_attrs(**kwargs)

class AirHeater(Heater):
    '''Air Heater class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'air heater',
        )
        self.set_attrs(**kwargs)

class WaterHeater(Heater):
    '''Water Heater class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'water heater',
        )
        self.set_attrs(**kwargs)

class CoolingTower(Generator):
    '''Cooling Tower class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'cooling tower',
        )
        self.set_attrs(**kwargs)

class HydrogenGenerator(Generator):
    '''Hydrogen Generator class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'hydrogen generator',
        )
        self.set_attrs(**kwargs)

class Electrolyzer(HydrogenGenerator):
    '''Electrolyzer class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'electrolyzer',
        )
        self.set_attrs(**kwargs)

class CombinedHeatPower(Generator):
    '''Combined Heat and Power class.

    ATTRIBUTES:
    state_space
    startup
    shutdown
    comm
    measure
    start_cost
    restart_time
    ramp_rate
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'combined heat power',
            source = 'ng',
            state_space = StateSpace(),
            startup = Startup(),
            shutdown = Shutdown(),
            comm = Comm(),
            measure = Measure(),
            start_cost = 0,
            restart_time = 0,
            ramp_rate = 0,
        )
        self.set_attrs(**kwargs)

class InternalCombustionEngine(CombinedHeatPower):
    '''Internal Combustion Engine class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'internal combustion engine',
        )
        self.set_attrs(**kwargs)

class FuelCell(CombinedHeatPower):
    '''Fuel Cell class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'fuel cell',
        )
        self.set_attrs(**kwargs)

class ReversibleFuelCell(FuelCell):
    '''Reversible Fuel Cell class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'reversible fuel cell',
        )
        self.set_attrs(**kwargs)

class MicroTurbine(CombinedHeatPower):
    '''Micro Turbine class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'micro turbine',
        )
        self.set_attrs(**kwargs)

class ElectricGenerator(Generator):
    '''Electric Generator class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'electric generator',
        )
        self.set_attrs(**kwargs)

class Renewable(Generator):
    '''Renewable class.

    ATTRIBUTES:
    us_state
    latitude
    longitude
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'renewable',
            source = 'renewable',
            us_state = 'CA',
            latitude = 0,
            longitude = 0,
        )
        self.set_attrs(**kwargs)

class Solar(Renewable):
    '''Solar generator class.

    ATTRIBUTES:
    tracking
    pv_type
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'solar',
            tracking = 'fixed',
            pv_type = 'flat',
        )
        self.set_attrs(**kwargs)

class Wind(Renewable):
    '''Wind generator class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'wind',
        )
        self.set_attrs(**kwargs)

class Hydro(Renewable):
    '''Hydro generator class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'hydro',
        )
        self.set_attrs(**kwargs)

class Storage(Component):
    '''Storage class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'storage',
        )
        self.set_attrs(**kwargs)

class ElectricStorage(Storage):
    '''Electric Storage class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'electric storage',
            source = 'electricity',
        )
        self.set_attrs(**kwargs)

class ThermalStorage(Storage):
    '''Thermal Storage class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'thermal storage',
        )
        self.set_attrs(**kwargs)

class HydrogenStorage(Storage):
    '''Hydrogen Storage class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'hydrogen storage',
        )
        self.set_attrs(**kwargs)

class HydroStorage(Storage):
    '''Hydro Storage class.

    No unique attributes.
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'hydro storage',
        )
        self.set_attrs(**kwargs)

class Utility(Component):
    '''Utility class.

    ATTRIBUTES:
    source
    size
    sum_start_month
    sum_start_day
    win_start_month
    win_start_day
    sum_rate_table
    win_rate_table
    sum_rates
    win_rates
    sellback_rate
    sellback_perc
    min_import_thresh
    ramp_rate
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'utility',
            source = 'electricity',
            size = 0,
            sum_start_month = 6,
            sum_start_day = 1,
            win_start_month = 10,
            win_start_day = 1,
            sum_rate_table = [],
            win_rate_table = [],
            sum_rates = [],
            win_rates = [],
            sellback_rate = -1,
            sellback_perc = 0,
            min_import_thresh = -float('inf'),
            ramp_rate = float('inf'),
        )
        self.set_attrs(**kwargs)

class DistrictHeat(Utility):
    '''District Heat class.

    ATTRIBUTES:
    capacity
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'district heat',
            capacity = float('inf'),
            output_fields = 'h',
        )
        self.set_attrs(**kwargs)

class DistrictCool(Utility):
    '''District Cool class.

    ATTRIBUTES:
    capacity
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'district cool',
            capacity = float('inf'),
            output_fields = 'c',
        )
        self.set_attrs(**kwargs)

class ACDCConverter(Component):
    '''AC/DC Converter class.

    ATTRIBUTES:
    source
    size
    ac_to_dc_eff
    dc_to_ac_eff
    capacity
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_attrs(
            name = 'ac dc converter',
            source = 'ac_dc',
            size = float('inf'),
            ac_to_dc_eff = 1,
            dc_to_ac_eff = 1,
            capacity = float('inf'),
        )
        self.set_attrs(**kwargs)
