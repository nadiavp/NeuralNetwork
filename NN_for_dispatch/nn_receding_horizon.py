from test_pytorch_nn import train_nn#, fire_nn
from test_pytorch_nn_sigmoid import train_nn_sigmoid, fire_nn
import torch
import xlrd
import xlsxwriter
import numpy as np
from numpy import degrees, radians, sin, cos, tan, arcsin, arccos
import datetime
import matplotlib as plt
import time
import os.path
####### dispatch the neural network in a receding horizon

def receding_horizon():
    ndisps = 365*24
    horizon = 24
    ngens = 14
    one_year_in = 365*24+1
    # load forecast data
    wb = xlrd.open_workbook('campus_testdata_2011_2014.xlsx')
    sheet = wb.sheet_by_index(0)
    inputs = torch.zeros(ndisps+horizon,4)#18
    renew_gen = torch.zeros(ndisps+horizon,1)

    for row in range(ndisps+horizon):
        r = row
        inputs[r, 0] = sheet.cell_value(row+1, 6) #E_demand
        inputs[r, 1] = sheet.cell_value(row+1, 7) #H_demand
        inputs[r, 2] = sheet.cell_value(row+1, 8) #C_demand
        inputs[r, 3] = sheet.cell_value(row+1, 11) #cost
        date = [datetime.datetime(int(sheet.cell_value(row+1,0)), 
            int(sheet.cell_value(row+1,1)), int(sheet.cell_value(row+1,2)),
            int(sheet.cell_value(row+1,3)), int(sheet.cell_value(row+1,4)), 
            int(sheet.cell_value(row+1,5)))]
        #renew_power = renewable_out(sheet.cell_value(row+1,10), date)
        #renew_gen[r, 0] = renew_power[0]
        renew_gen[r,0] = sheet.cell_value(row+1,12) #solar generation
        inputs[r,0] = inputs[r,0]-renew_gen[r,0]

    #inputs[:,0] -= renew_gen

    #read in initial condition
    wb = xlrd.open_workbook('Campus_MI_18component.xlsx')
    sheet = wb.sheet_by_index(0)
    IC = torch.zeros(horizon,14)
    for row in range(one_year_in,one_year_in+horizon):
        r = row-365*24-1
        IC[r,0] = sheet.cell_value(row,2)/7000#GT1
        IC[r,1] = sheet.cell_value(row,3)/5000#GT2
        IC[r,2] = sheet.cell_value(row,4)/2000#FC1
        IC[r,3] = sheet.cell_value(row,5)/2000#FC2
        IC[r,4] = sheet.cell_value(row,6)/500#sGT
        IC[r,5] = sheet.cell_value(row,7)/1500#Diesel
        IC[r,6] = sheet.cell_value(row,8)/20000#Heater
        IC[r,7] = sheet.cell_value(row,9)/10000#chiller1
        IC[r,8] = sheet.cell_value(row,10)/10000#chiller2
        IC[r,9] = sheet.cell_value(row,11)/7500#small Chiller1
        IC[r,10] = sheet.cell_value(row,12)/7500#small Chiller2
        IC[r,11] = sheet.cell_value(row,13)/30000#battery
        IC[r,12] = sheet.cell_value(row,14)/75000#hot water tank
        IC[r,13] = sheet.cell_value(row,15)/200000#cold water tank    
        
    # train NN
    layers = 7
    tic = time.time()
    model, acc, acc_test, input_scale_factors, output_scale_factors, loss_rate = train_nn_sigmoid(layers)
    toc = time.time() - tic
    print('time for training: '+str(toc))
    #record loss function vs. iterations, training accuracy, test accuracy
    workbook = xlsxwriter.Workbook('Training_NN_07_s_01_noIC.xlsx')
    worksheet = workbook.add_worksheet()
    worksheet.write(0,0,toc) #time for training and validation testing
    for row in range(len(acc)):
        worksheet.write(row,1,acc[row])
        worksheet.write(row,2,acc_test[row])
    for row in range(len(loss_rate)):
        worksheet.write(row,3,loss_rate[row])
    workbook.close()

    #scale inputs 
    scaled_inputs = inputs
    #scaled_inputs[:24,4:] = IC[:24,:]
    for row in range(ndisps+horizon):
        for i in range(4):
            scaled_inputs[row,i] = inputs[row,i]/input_scale_factors[i]

    # run receding horizon dispatch
    tic = time.time()
    outputs = np.zeros((ndisps*horizon, 14))
    solar_gen = np.zeros((ndisps*horizon, 1))
    t = 0
    while t < ndisps:  
        # solve nn for all timesteps in horizon
        disp = fire_nn(model,scaled_inputs[t:t+horizon,:])
        #update initial conditions
        #scaled_inputs[t+1:t+horizon+1,4:] = disp
        #record horizon disp, and implemented first timestep disp
        outputs[t*24:(t+1)*24] = disp.detach().numpy()*output_scale_factors
        solar_gen[t*24:(t+1)*24,0] = renew_gen[t:t+horizon,0].detach().numpy()
        t=t+1

    toc = time.time()-tic
    print('time for receding horizon: '+str(toc))

    #write to excel's
    workbook = xlsxwriter.Workbook('Dispatch_07_s_01_noIC.xlsx')
    worksheet = workbook.add_worksheet()
    for row in range(len(outputs[:,0])):
        for col in range(ngens):
            worksheet.write(row, col, outputs[row,col])
        worksheet.write(row, ngens+1, solar_gen[row,0])
    worksheet2 = workbook.add_worksheet()
    for row in range(len(scaled_inputs[:,0])):
        for col in range(len(input_scale_factors)):
            worksheet2.write(row, col, scaled_inputs[row, col].detach().numpy()*input_scale_factors[col])
        worksheet2.write(row, 4, renew_gen[row])
        # for col in range(5,ngens+4):
        #     worksheet2.write(row, col, scaled_inputs[row,col].detach().numpy()*output_scale_factors[col-4])
    workbook.close()



def renewable_out(irrad, date):
    latitude = 40
    longitude = -105
    time_zone = -6
    _, _, azimuth, zenith = solar_calc(longitude, latitude, time_zone, date)
    power = 40000*irrad/1000*np.degrees(np.cos(zenith-40.9524))*max(0, (np.degrees(np.cos(azimuth-180)))) * 0.19
    return power







def solar_calc(longitude, latitude, time_zone, inp_date):
    ''' Calculate position of sun and sunrise and sunset times.\n
    Calculated using NOAA solar calculations available at:
    https://www.esrl.noaa.gov/gmd/grad/solcalc/NOAA_Solar_Calculations_day.xls \n
    LONGITUDE is longitude (+ to east).\n
    LATITUDE is latitude (+ to north).\n
    TIME_ZONE is the time zone (+ to east).\n
    INP_DATE is the date as a list of datetime objects, i.e. Jan 1 2017 = 
    datetime(2017,1,1).\n
    SUNRISE and SUNSET are given in fraction of the day, i.e. 6am = 6/24.\n
    AZIMUTH and ZENITH are given in degrees.'''

    tpm = np.array([(x - x.replace(hour=0, minute=0, second=0)).seconds / 24 / 3600 \
        for x in inp_date]) # time past midnight [days]
    jd = [x.toordinal() + 0.5 + y - time_zone/24 for x, y in zip(inp_date, tpm)]
        # Julian day [days]
        # Julian calendar epoch: Jan 1, 4713 B.C., 12:00:00.0
        # UNIX epoch: Jan 1, 0001 A.D., 00:00:00.0
        # MATLAB epoch: Jan 1, 0000 A.D., 00:00:00.0
        # 1721423.5 days from Julian epoch to UNIX epoch, plus 1 to match
        #     Microsoft Excel's year 1900 exception to the Gregorian calendar.
        # USNO Julian Date Converter:
        #     http://aa.usno.navy.mil/data/docs/JulianDate.php
        # More info on Excel dates:
        #     http://calendars.wikia.com/wiki/Microsoft_Excel_day_number
    jc = (np.array(jd) - 2451545) / 36525 # Julian century
    geom_mean_long_sun = (280.46646 + jc * (36000.76983 + jc*0.0003032)) % 360
        # [degrees]
    geom_mean_anom_sun = 357.52911 + jc * (35999.05029 - 0.0001537*jc) # [degrees]
    eccent_earth_orbit = 0.016708634 - jc * (0.000042037 + 0.0000001267*jc)
    sun_eq_of_center = \
        sin(radians(geom_mean_anom_sun)) * (1.914602 - jc * (0.004817 + 0.000014*jc)) \
        + sin(radians(2*geom_mean_anom_sun)) * (0.019993 - 0.000101*jc) \
        + sin(radians(3*geom_mean_anom_sun)) * 0.000289
    sun_true_long = geom_mean_long_sun + sun_eq_of_center # [degrees]
    mean_obliq_ecliptic = 23 + (26 + (21.448 - jc * (46.815 \
        + jc * (0.00059 - jc*0.001813))) / 60) / 60 # [degrees]
    sun_app_long = sun_true_long - 0.00569 \
        - 0.00478 * sin(radians(125.04 - 1934.136*jc)) # [degrees]
    obliq_corr = mean_obliq_ecliptic \
        + 0.00256*cos(radians(125.04 - 1934.136*jc)) # [degrees]
    sun_declin = degrees(arcsin(sin(radians(obliq_corr)) \
        * sin(radians(sun_app_long)))) # solar declination [degrees]
    var_y = tan(radians(obliq_corr/2)) * tan(radians(obliq_corr/2))
    eq_of_time = 4 * degrees( \
        var_y * sin(2*radians(geom_mean_long_sun)) \
        - 2 * eccent_earth_orbit * sin(radians(geom_mean_anom_sun)) \
        + 4 * eccent_earth_orbit * var_y * sin(radians(geom_mean_anom_sun)) * cos(2*radians(geom_mean_long_sun)) \
        - 0.5 * var_y * var_y * sin(4*radians(geom_mean_long_sun)) \
        - 1.25 * eccent_earth_orbit * eccent_earth_orbit * sin(2*radians(geom_mean_anom_sun)))
    ha_sunrise = degrees(arccos( \
        cos(radians(90.833)) / (cos(radians(latitude)) * cos(radians(sun_declin))) \
        - tan(radians(latitude)) * tan(radians(sun_declin))))
        # sunlight hours [degrees]
    solar_noon = (720 - 4*longitude - eq_of_time + time_zone*60) / 1440
    sunrise = solar_noon - ha_sunrise * 4 / 1440 # Local Sidereal Time (LST)
    sunset = solar_noon + ha_sunrise * 4 / 1440 # Local Sidereal Time (LST)
    tst = (tpm*1440 + eq_of_time + 4*longitude - 60*time_zone) % 1440 # True Solar Time [min]
    hour_angle = [x/4 + 180 if x/4 < 0 else x/4 - 180 for x in tst] # [degrees]
    zenith = degrees(arccos( \
        sin(radians(latitude)) * sin(radians(sun_declin)) \
        + cos(radians(latitude)) * cos(radians(sun_declin)) * cos(radians(hour_angle))))  # [degrees]
    ang = degrees(arccos( \
        ((sin(radians(latitude))*cos(radians(zenith))) - sin(radians(sun_declin))) \
        / (cos(radians(latitude))*sin(radians(zenith)))))
    azimuth = np.array([(x+180) % 360 if y > 0 else (540-x) % 360 \
        for x, y in zip(ang, hour_angle)])
        # [degrees] (clockwise from N)
    return sunrise, sunset, azimuth, zenith


receding_horizon()