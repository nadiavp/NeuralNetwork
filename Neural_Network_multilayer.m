classdef Neural_Network_multilayer
    properties
        %define structure parameters
        inputLayerSize
        outputLayerSize
        nhiddenLayers%number of hiddenlayers, layer size is determined automatically
        %define wieght parameters
        Wlayer
        %define bias
        blayer
        %define scale factor for how complex you want to allow your system
        %to be
        lambda
        %is it a classification network (generators off/on)
        classify
        %constant for node function
        nodeconst
        %madtrain is a constant for what size steps and how many iterations
        %to train to
        maxtrain
    end
    methods
        function obj = Neural_Network_multilayer(inputLayerSize, outputLayerSize, nhiddenLayers, varargin)
            %% initialization:
            %one weight per input per node, one bias per node
            %all the inputs 
            if isnumeric(inputLayerSize)
                obj.inputLayerSize = inputLayerSize;
            end
            %hidden layers should start with a number of nodes between the
            %nodes above and below them, for example: if 7 inputs and 3
            %and 3 hidden layers, the hidden layers have 5,4,5 nodes
            %respectively
            nodesPerLayer = [inputLayerSize;outputLayerSize];
            if isnumeric(nhiddenLayers)
                obj.nhiddenLayers = nhiddenLayers;
                if nhiddenLayers>0 %if you have hidden layers, set up the number of nodes, then set up the weight and bias matrices
                    nodesPerLayer = ceil(linspace(inputLayerSize,outputLayerSize,nhiddenLayers+2));
                    for i = 1:1:nhiddenLayers
                        obj.blayer{i} = rand(1,nodesPerLayer(i+1));
                    end
                end
                %weights occur along branches between nodes (between biases)
                for i = 1:1:nhiddenLayers+1
                    obj.Wlayer{i} = rand(nodesPerLayer(i), nodesPerLayer(i+1));
                end
            end            
            %the output layer has connections from every previous node to
            %every output
            if isnumeric(outputLayerSize)
                obj.blayer{end+1} = rand(1,outputLayerSize);
                obj.outputLayerSize = outputLayerSize;
            end     

            %default conditions
            obj.lambda = .0001;
            obj.classify = false;
            obj.nodeconst = 1;
            obj.maxtrain = 1;
            
            
            if length(varargin)==1%the first input is if it is a classification network, then node const, then lambda
               classify = iscellstr(varargin);
               lambda = isnumeric(varargin);
               obj.lambda = varargin(lambda);
               obj.classify = (nnz(classify)>0);
            elseif length(varargin)==2
                obj.classify = strcmp(varargin{1},'classify');
                obj.lambda = varargin{2};
            end
        end
        
        function output = forward(self, X)
            %forward propogate inputs (X) the whole way through the
            %network, this is not very useful for trainning, but gives the
            %whole network result
            for i = 1:1:length(self.Wlayer)
                output = forward1step(self,X,i);
                X = output;
            end
        end
        
        function yHat = forward1step(self,X,step)
            %propogate forward only one step, this is useful for training
%             if step ==1
%                 yHat = X.*(ones(length(X(:,1)),1)*self.Wlayer{1}') + ones(length(X(:,1)),1)*self.blayer{1}';
%             else
%                 yHat = (X*self.Wlayer{step} + ones(length(X(:,1)),1)*self.blayer{step}');
%             end
            yHat = X*self.Wlayer{step}+ones(length(X(:,1)),1)*self.blayer{step};
            yHat = activationf(self, yHat); %use the activation function scaled by 1
        end
        
        function a2 = activationf(self,yHat)
            %apply activation function
            %if it is a classifyer use a sigmoid function theta(s) =
            %exp(s)/(1+s)
%             if self.classify
                a2 = exp(self.nodeconst*yHat)./(1+exp(self.nodeconst*yHat));
                if nnz(isnan(a2))>0
                    a2(and(isnan(a2),yHat>0)) = 1;%if numbers are too big inf/inf, make a2=1
                    a2(isnan(a2)) = 0;%if numbers are to negative -inf/-inf, make a2=0
                end
%             else %if it is a numeric output network don't include activation
%                 a2 = yHat*self.nodeconst;
%             end
        end
    end
end
