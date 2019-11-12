#pytorch tests
import numpy as np 
import xlrd
import torch

def train_nn(layers=4):
    dtype = torch.float
    #device = torch.device("cpu")
    device = torch.device("cuda:0")


    load_e = []
    load_h = []
    load_c = []
    t_db = []
    irrad = []

    wb = xlrd.open_workbook('Campus_loads_weather.xlsx')
    sheet = wb.sheet_by_index(0)
    inputs= torch.zeros(sheet.nrows-3,18)#, device=device, dtype=dtype)
    e_dem_max = sheet.cell_value(1,7)
    h_dem_max = sheet.cell_value(1,8)
    c_dem_max = sheet.cell_value(1,9)
    cost_max = sheet.cell_value(1,12)
    for row in range(2,sheet.nrows-1):
        inputs[row-2,0] = sheet.cell_value(row,1)#E_dem
        inputs[row-2,1] = sheet.cell_value(row,2)#H_dem
        inputs[row-2,2] = sheet.cell_value(row,3)#C_dem
        inputs[row-2,3] = sheet.cell_value(row,4)/cost_max #utility electric cost
        #inputs[row-2,3] = sheet.cell_value(row,4)/sheet.cell_value(1,10)#Temp_db_C
        #inputs[row-2,4] = sheet.cell_value(row,5)/sheet.cell_value(1,11)#Direct_normal_irradiance
        # load_e.append(row)
        # load_h.append(row[1])
        # load_c.append(row[2])
        # t_db.append(row[3])
        # irrad.append(row[4])
        # inputs.append(row)
    #inputs[:,1] = (inputs[:,1]-min(inputs[:,1]))/h_dem_max
    #inputs[:,2] = (inputs[:,2]-min(inputs[:,2]))/c_dem_max
    inputs[:,1] = inputs[:,1]/h_dem_max
    inputs[:,2] = inputs[:,2]/c_dem_max


    print('inputs read')

    wb = xlrd.open_workbook('Campus_MI_18component.xlsx')
    sheet = wb.sheet_by_index(0)
    disp = torch.zeros(sheet.nrows-2,14)#, device=device, dtype=dtype)
    for row in range(1,sheet.nrows-1):
        r = row-1
        inputs[r,0] -=sheet.cell_value(row,1)
        disp[r,0] = sheet.cell_value(row,2)/7000#GT1
        disp[r,1] = sheet.cell_value(row,3)/5000#GT2
        disp[r,2] = sheet.cell_value(row,4)/2000#FC1
        disp[r,3] = sheet.cell_value(row,5)/2000#FC2
        disp[r,4] = sheet.cell_value(row,6)/500#sGT
        disp[r,5] = sheet.cell_value(row,7)/1500#Diesel
        disp[r,6] = sheet.cell_value(row,8)/20000#Heater
        disp[r,7] = sheet.cell_value(row,9)/10000#chiller1
        disp[r,8] = sheet.cell_value(row,10)/10000#chiller2
        disp[r,9] = sheet.cell_value(row,11)/7500#small Chiller1
        disp[r,10] = sheet.cell_value(row,12)/7500#small Chiller2
        disp[r,11] = sheet.cell_value(row,13)/30000#battery
        disp[r,12] = sheet.cell_value(row,14)/75000#hot water tank
        disp[r,13] = sheet.cell_value(row,15)/200000#cold water tank

    inputs[:,0] = inputs[:,0]/e_dem_max#inputs[:,0] = (inputs[:,0]-min(inputs[:,0]))/max(inputs[:,0])
    inputs[1:,4:] = disp[:-1,:]
    inputs = inputs[1:,:]
    disp = disp[1:,:]
    print('outputs read')

    #shuffle and separate training from testing
    shuffled = [inputs,disp]
    shuffled = torch.randperm(len(inputs[:,0]))
    split_line = int(np.floor(len(inputs[:,0])/10))
    inputs = inputs[shuffled,:]
    disp = disp[shuffled,:]
    inputs_train = inputs[:-split_line,:]
    inputs_test = inputs[-split_line:, :]
    disp_train = disp[:-split_line, :]
    disp_test = disp[-split_line:, :]
    disp = disp_train
    inputs = inputs_train


    #batch size, input dimension hidden dimension, output dimension
    N, D_in, H, D_out = len(inputs[:,0]), len(inputs[0,:]), int(np.ceil(len(inputs[0,:])+len(disp[0,:])/2)), len(disp[0,:])
    H2 = int(np.round(H*1.2))
    H3 = int(np.round(H*0.8))
    H4 = H2

    x = inputs
    y = disp

    if layers == 1:
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, D_out)
        )
    elif layers == 2:
        model = torch.nn.Sequential(
        torch.nn.Linear(D_in,H),
        torch.nn.ReLU(),
        torch.nn.Linear(H,D_out) 
        )
    elif layers == 3:
        model = torch.nn.Sequential(
        torch.nn.Linear(D_in,H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
        )
    elif layers == 4:
        model = torch.nn.Sequential(
        torch.nn.Linear(D_in,H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H2),
        torch.nn.ReLU(),
        torch.nn.Linear(H2, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
        )
    elif layers == 5:
        model = torch.nn.Sequential(
        torch.nn.Linear(D_in,H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H2),
        torch.nn.ReLU(),
        torch.nn.Linear(H2, H3),
        torch.nn.ReLU(),
        torch.nn.Linear(H3, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
        )
    elif layers == 6:
        model = torch.nn.Sequential(
        torch.nn.Linear(D_in,H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H2),
        torch.nn.ReLU(),
        torch.nn.Linear(H2, H3),
        torch.nn.ReLU(),
        torch.nn.Linear(H3, H2),
        torch.nn.ReLU(),
        torch.nn.Linear(H2, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out)
        )
    else: #layers = 7
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in,H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, H2),
            torch.nn.ReLU(),
            torch.nn.Linear(H2, H3),
            torch.nn.ReLU(),
            torch.nn.Linear(H3, H4),
            torch.nn.ReLU(),
            torch.nn.Linear(H4, H2),
            torch.nn.ReLU(),
            torch.nn.Linear(H2, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out)
        )


    loss_fn = torch.nn.MSELoss(reduction='sum')
    n_iters = 50000
    loss_rate = np.zeros(n_iters)

    learning_rate = 1e-6
    for t in range(n_iters):
        y_pred = model(x)

        loss = loss_fn(y_pred, y)
        print(t, loss.item())
        loss_rate[t] = loss.item()

        model.zero_grad()

        loss.backward()

        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate *param.grad

    y_pred_np = y_pred.detach().numpy()
    y_np = y.detach().numpy()
    acc = sum((y_pred_np-y_np)**2)/len(y[:,0])
    #a = 0

    #test
    y_pred = model(inputs_test)
    y_pred_test = y_pred.detach().numpy()
    y_test = disp_test.detach().numpy()
    acc_test = sum((y_pred_test-y_test)**2)/len(y_test[:,0])
    #b=0

    input_scale_factors = np.array([e_dem_max, h_dem_max, c_dem_max, cost_max])
    output_scale_factors = np.array([7000, 5000, 2000, 2000, 500, 1500, 20000, 10000, 10000, 7500, 7500, 30000, 75000, 200000])

    return model, acc, acc_test, input_scale_factors, output_scale_factors, loss_rate

def fire_nn(model, scaled_inputs):
    y_pred = model(scaled_inputs)
    return y_pred
    
