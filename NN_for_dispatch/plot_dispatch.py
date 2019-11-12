import numpy as np 
import matplotlib.pyplot as plt 

def plot_dispatch(dispatch, demand):
    ngens = len(dispatch[0,:])
    chillers = [7,8,9,10]
    heaters = [6]
    gens = [0,1,2,3,4,5]
    cold_storage = [13]
    hot_storage = [12]
    e_storage = [11]
    horizon = len(dispatch[:,0])
    ind = np.arange(horizon)
    time_label = [str(i) for i in ind]
    width = 1

    #figure out what is imported from the utility to campus
    generate = np.zeros((horizon,1))
    for i in gens:
        generate += dispatch[:,i]
    for i in e_storage:
        generate[1:] = generate[1:] + (dispatch[1:,i] - dispatch[:-1,i])
    utility = demand-generate

    #electric plot
    for i in gens:
        p1 = plt.bar(ind, dispatch[:,i], width)
    for i in e_storage:
        p2 = plt.plot(ind, dispatch[:,i])
        p3 = plt.bar(ind[1:], dispatch[1:,i]-dispatch[:-1,i])
    p4 = plt.bar(ind, utility, width)
    plt.ylabel('Generation (kW)')
    plt.title('NN Electric Dispatch')
    plt.xticks(ind, time_label)
    plt.xlabel('Time (hrs)')
    plt.show()

    #cooling plot
    for i in chillers:
        p1 = plt.bar(ind, dispatch[:,i], width) 
    for i in cold_storage:
        p2 = plt.plot(ind, dispatch[i,:])
        p3 = plt.bar(ind[1:], dispatch[i,1:]-dispatch[i,:-1])
    plt.ylabel('Generation (kW)')
    plt.title('NN Cooling Dispatch')
    plt.xticks(ind, time_label)
    plt.xlabel('Time (hrs)')
    plt.show()

    #heat plot
    for i in heaters:
        p1 = plt.bar(ind, dispatch[:,i], width) 
    for i in hot_storage:
        p2 = plt.plot(ind, dispatch[i,:])
        p3 = plt.bar(ind[1:], dispatch[i,1:]-dispatch[i,:-1])
    plt.ylabel('Generation (kW)')
    plt.title('NN Heat Dispatch')
    plt.xticks(ind, time_label)
    plt.xlabel('Time (hrs)')
    plt.show()
