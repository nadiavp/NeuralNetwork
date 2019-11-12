#pytorch tests
import numpy as np 
import xlrd
import torch

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
inputs= torch.zeros(sheet.nrows-3,5, device=device, dtype=dtype)
for row in range(2,sheet.nrows-1):
    inputs[row-2,0] = sheet.cell_value(row,1)/sheet.cell_value(1,6)#E_dem
    inputs[row-2,1] = sheet.cell_value(row,2)/sheet.cell_value(1,7)#H_dem
    inputs[row-2,2] = sheet.cell_value(row,3)/sheet.cell_value(1,8)#C_dem
    inputs[row-2,3] = sheet.cell_value(row,4)/sheet.cell_value(1,9)#Temp_db_C
    inputs[row-2,4] = sheet.cell_value(row,5)/sheet.cell_value(1,10)#Direct_normal_irradiance
    # load_e.append(row)
    # load_h.append(row[1])
    # load_c.append(row[2])
    # t_db.append(row[3])
    # irrad.append(row[4])
    # inputs.append(row)

print('inputs read')

wb = xlrd.open_workbook('Campus_MI_18component.xlsx')
sheet = wb.sheet_by_index(0)
disp = torch.zeros(sheet.nrows-2,14, device=device, dtype=dtype)
for row in range(1,sheet.nrows-1):
    r = row-1
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
    disp[r,13] = sheet.cell_value(row,15)/20000#cold water tank

print('outputs read')


# disp = []
# with open('Campus_MI_18component.xlsx') as csvfile:
#     dispreader = csv.reader(csvfile)
#     for row in loadreader:
#         disp.append(row)

#batch size, input dimension hidden dimension, output dimension
N, D_in, H, D_out = len(inputs[:,0]), len(inputs[0,:]), int(np.ceil(len(inputs[0,:])+len(disp[0,:])/2)), len(disp[0,:])

x = inputs
y = disp
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w3 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H,H, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(100000):
    #forward:
    y_pred = x.mm(w1).clamp(min=0).mm(w2).clamp(min=0).mm(w3)
    #Forward pass: compute predicted dispatch
    # h = x.mm(w1)
    # h_relu = h.clamp(min=0)
    # y_pred = h_relu.mm(w2)

    # #compute and print loss
    # loss = (y_pred-y).pow(2).sum().item()
    # print(t,loss)
    loss = (y_pred-y).pow(2).sum()
    print(t, loss.item())

    # #Backprop to compute gradients of w1 and w2 with respect to loss
    # grad_y_pred = 2.0*(y_pred-y)
    # grad_w2 = h_relu.t().mm(grad_y_pred)
    # grad_h_relu = grad_y_pred.mm(w2.t())
    # grad_h = grad_h_relu.clone()
    # grad_h[h <0] =0
    # grad_w1 = x.t().mm(grad_h)
    loss.backward()

    # #update weights
    # w1 -= learning_rate * grad_w1
    # w2 -= learning_rate*grad_w2
    with torch.no_grad():
        w1 -= learning_rate*w1.grad
        w2 -= learning_rate*w2.grad
        w3 -= learning_rate*w3.grad

        #manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()
        w3.grad.zero_()

