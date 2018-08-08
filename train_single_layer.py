def train_single_layer(net, desired_out, inputs):
    #this does forward and backward propagation for a one layer network for a set of
    #generators and a demand
    #inputs: 
    #net        - instance of class NueralNetwork
    #desired_out - desired network output when using forward funnction in the form of a vertical vector, 
    #inputs     - matrix of inputs of size inputlength x number of outputs
    #outputs:
    #net        - trained neural network
    #sqrerror   - square error for each training example
    #train_error - vector of mean square error at each iteration

    #initialize
    [sqrerror, dedw, dedb] = finderror(net, inputs, desired_out)
    tolerance = 0.0001
    iterations = 0
    laststep = np.zeros(np.shape(dedw))
    lastbstep = np.zeros(np.shape(dedb))
    a = 1
    momentum = 0.25
    #train until you meet the threshold or the number of iterations
    while np.count_nonzeros(abs(sqrerror)>tolerance)>0:
        #find error and relation to weights and biases
        [sqrerror, dedw, dedb] = finderror(net, inputs, desired_out)
        step = dedw*a/100 + laststep*momentum/100
        bstep = dedb*a/100 + lastbstep*momentum/100

        net.wlayer1 = net.wlayer1 - step
        net.blayer1 = net.blayer1 - bstep

        #check error with new weight and bias
        [sqrerrornew, _, _] = finderror(net, inputs, desired_out)

        #if it gets worse, try a different step size
        if sum(sum(abs(sqrerrornew)))>= sum(sum(abs(sqrerror))) or np.count_nonzeros(sqrerrornew==float('inf')):
            if abs(a) < 1e-12:
                a = 1
            else:
                a = a/10

            #undo the last change
            net.wlayer1 = net.wlayer1+step
            net.blayer1 = net.blayer1+bstep
            laststep = np.zeros(np.shape(laststep))
            lastbstep = np.zeros(np.shape(lastbstep)) 

        #if it gets better, keep the change and keep going
        else:
            laststep = step
            lastbstep = bstep
            sqrerror = sqrerrornew
            #if you are below tolerance, stop
            if np.count_nonzeros(abs(sqrerrornew)>tolerance) == 0:
                print('below tolerance')
                break

        #if you have hit your max iterations, stop
        if iterations>1e4:
            print('not converging after 10^4 iterations, exiting training loop')
            sqrerror = sqrerrornew
            break

        iterations = iterations+1


    return net, sqrerror



def finderror(net, inputs, desired_out):
    #this is a sub function of train_single_layer which calcualtes the error in the 
    #network output and also finds the necessary change in wieghts and biases
    #inputs:
    #net        - instance of class NueralNetwork in the process of being trained
    #inputs     - training examples in matrix
    #desired_out    - training example desired outputs in matrix
    #outputs:
    #derrordw   - derivative of weight with respect to error
    #derrordb   - derivative of bias with respect to error
    #cost       - 1/2 square error for each training example

    net_out = net.forward(inputs)
    error = desired_out-net_out
    cost = error**2 *0.5
    if net.classify: #if it has a sigmoid function
        derrordw = -2*np.transpose(inputs)*error*(net_out*(1-net_out))*net.nodeconst/len(desired_out[:,0])
        derrordb = -2*sum(error*net_out*(1-net_out)*net.nodeconst)/len(desired_out[:,0])
    else: #if no activation function
        derrordw = np.transpose(-error*inputs)
        derrordb = -1/len(error[0,:]) * sum(error,axis=2)

    return cost, derrordw, derrordb