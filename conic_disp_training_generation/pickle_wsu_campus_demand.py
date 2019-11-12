import os
import pickle
import xlrd
#os.chdir('C:/Users/MME-Admin/Documents/Github/EAGERS_py')
import numpy as np
from class_definition.test_data import (TestData,Demand,Weather)
from instance.create_timestamp import create_timestamp

def load_demand():
    testdata = TestData()

    wb = xlrd.open_workbook('instance\wsu_campus_2009_2012_irrad_fix.xlsx')
    dem_sheet = wb.sheet_by_index(0)
    weather_sheet = wb.sheet_by_index(1)
    e = []
    h = []
    c = []

    tdb = []#dry bulb temp
    irrad_dire_norm = []#direct normal irradiation
    for i in range(1,83825):
        e.append(dem_sheet.cell_value(i,0))
        h.append(dem_sheet.cell_value(i,1))
        c.append(dem_sheet.cell_value(i,2))

        tdb.append(weather_sheet.cell_value(i,0))
        irrad_dire_norm.append(weather_sheet.cell_value(i,1))

    #heat has way more nans, so remove them
    h = h[:37810]

    demand = Demand()
    demand.e = [np.array(e)*32152/117415.7,np.array(e)*53568.5/117415.7,np.array(e)*38480/117415.7,np.array(e)*3215.2/117415.7]
    demand.h = [np.array(h)*32152/117415.7, np.array(h)*53568.5/117415.7, np.array(h)*38480/117415.7, np.array(h)*3215.2/117415.7]
    demand.c = [np.array(c)*32152/117415.7, np.array(c)*53568.5/117415.7, np.array(c)*38480/117415.7, np.array(c)*3215.2/117415.7]

    weather = Weather()
    weather.t_db = tdb
    weather.irrad_dire_norm = irrad_dire_norm

    testdata.demand = demand
    testdata.weather = weather


    timestamp = create_timestamp(2009,12,3,2,45,83825,dt=.25)
    testdata.timestamp = timestamp

    # file_Name = "wsu_campus_demand_2009_2012"
    # fileObject = open(file_Name,'wb')
    # pickle.dump(testdata,fileObject, protocol=2)
    # fileObject.close()
    return testdata



# pickle_demand()