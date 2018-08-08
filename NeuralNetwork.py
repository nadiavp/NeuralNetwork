class NeuralNetwork:
    import numpy as np
    #class definition for single layer neural NeuralNetwork
    def __init__(self,input_layer_size, output_layer_size):
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.lambda = 0.0001
        self.classify = True
        self.nodeconst = 1
        self.avrginput = []
        self.stddev = []

        #initialize the weights and biases
        self.wlayer1 = rand(input_layer_size,output_layer_size)
        self.blayer1 = rand(1,output_layer_size)

    def forward(self, x):
        #foraward propogation function where self is the neural network and x are inputs 
        if sum(np.shape(np.transpose(x))==np.shape(self.wlayer1))==2:#if input format allows direct multiplication
            z2 = x*np.transpose(self.wlayer1)
            y_hat = sum(z2, axis=2)+np.transpose(self.blayer1)

        else: #if only one row of inputs, or one row of inputs per timestep
            y_hat = x*self.wlayer1 + self.blayer1

        return y_hat

    def activationf(self, y_hat):
        #apply activation function
        #if it is a classifyer use a sigmoid function
        if self.classify:
            a2 = 1/(1+exp(-self.nodeconst*y_hat))
            if np.count_nonzeros(np.isnan(a2)) >0:
                a2(np.isnan(a2) and y_hat>0) = 1 #if numbers are too big inf/inf, make a2 = 1
                a2(np.isnan(a2)) = 0 #if numbers are negative -inf/-inf, make a2 = 0

        else:#if there is no activation function
            a2 = y_hat*self.nodeconst

        return a2
    