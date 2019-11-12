import sys
import pickle
import datetime
from datetime import timedelta, datetime
import numpy as np
from numpy import genfromtxt
sys.path.append('C:/Users/MME-Admin/Documents/GitHub/EAGERS_py')
from class_definition.test_data import (TestData,Demand,Weather)
from dev.tools.mat_to_py import datenum_to_datetime


test = genfromtxt('C:/Users/MME-Admin/Documents/GitHub/EAGERS_py/instance/Test_Data.csv', delimiter = ',')

e = test[:,0,None]
h = test[:,1, None]
t_db = test[:,2,None]
irrad_dire_norm = test[:,3,None]
ts = test[:,4,None]

TestData = TestData()

demand = Demand()
demand.e = e
demand.h = h
TestData.demand = demand

weather = Weather()
weather.t_db = t_db
weather.irrad_dire_norm = irrad_dire_norm
TestData.weather = weather

timestamp = datetime(2007,10,1) + np.arange(24*31)*timedelta(hours = 1)
TestData.timestamp = timestamp

file_Name = "test_single.pickle"
fileObject = open(file_Name,'wb')
pickle.dump(TestData,fileObject)
fileObject.close()
