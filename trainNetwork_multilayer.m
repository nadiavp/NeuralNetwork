function [Net,sqrerror] = trainNetwork_multilayer(Net,desiredOut, inputs, batch, varargs)
%this does forward propagation for a one layer network for a set of
%generators and a demand
%inputs: network, desiredOut: desired network output when using forward
%funnction in the form of a vertical vector, inputs: matrix of inputs of
%size inputlength x number of outputs
%inputs in order: ub, lb, f, H for each generator, demand, $/kWgrid
%batch: boolean indicating if you are doing batch learning or not
%if batch is true, varargs indicates the iteration you are on, if it is
%false, then varargs indicates the max iterations allowed



[sqrerror,dedW,dedb] = finderror(Net,inputs,desiredOut,length(Net.Wlayer));%find the error and the gradient of the error for the last layer



%% use BFGS technique to train using derrordW
tolerance = .0001;%.*ones(1,length(sqrerror(1,:)));
layer = Net.nhiddenLayers + 1;
if batch
    max_iters = varargs;
    a = 1;
    iterations = 0;
else
    max_iters = 10;
    a = 1;
    iterations = rem(varargs,max_iters);
    if varargs>max_iters
        layer = layer-floor(rem((varargs/max_iters),layer));
    end
end



laststep = zeros(size(Net.Wlayer{layer}));
lastbstep = zeros(size(Net.blayer{layer}));
momentum = 0.25;
des_out_layer = desiredOut;
%     a = 100/(100^Net.classify);%start at 100 for numeric, start at 1 for classification
while nnz(sqrerror>tolerance)>0 %keep training until you get the desired output
    %find error and relation to weights and biases
    [sqrerror, dedW, dedb] = finderror(Net, inputs, des_out_layer, layer);
    iterations = iterations+1;
    wstep = dedW.*a/100 + laststep.*momentum/100;%training for weights
    bstep = dedb.*a/100 + lastbstep.*momentum/100;%training step for bias
    %obtain training step direction
    %perform line search to find acceptable scalar for step size
    %must find scalar that minimizes error from Wlayer1+scalar*trainingstep
    %solve 0 = sqrerror(Wlayer1+scalar*trainingstep), or require sufficient
    %decrease in error
%     if Net.classify
%         momentum = .1;%.3 is too high, .1 is too low, .2 does well for test2E_1BS
%     else momentum = 0.4;
%     end
    Net.blayer{layer} = Net.blayer{layer}-bstep;
    Net.Wlayer{layer} = Net.Wlayer{layer}-wstep;
    
    %check error with new weight and bias
    [sqrerrornew, ~, ~] = finderror(Net, inputs, des_out_layer,layer);

    if sum(sum(abs(sqrerrornew)))>=sum(sum(abs(sqrerror))) || nnz(isinf(sqrerrornew))>0%|| nnz(isnan(dedWnew))>0%if the error gets worse or you have reached a flat point
        if abs(a) <1e-12 %try different size steps
            a = 1;
        else
            a = a/10;
        end
        Net.blayer{layer} = Net.blayer{layer}+bstep;
        Net.Wlayer{layer} = Net.Wlayer{layer}+wstep;
        laststep = zeros(size(laststep));
        lastbstep = zeros(size(lastbstep));
    else %if it gets better, keep going
        laststep = wstep;
        lastbstep = bstep;
        if nnz(sqrerrornew>tolerance)==0
%                 disp('below tolerance');
            layer = layer-1; %if you are below tolerance go up a layer
            if layer == 0
                break
            end
        end
        sqrerror = sqrerrornew;
    end
    if iterations>max_iters%*length(desiredOut(:,1))/100%10^4 iterations per 100 timesteps
%             disp('not converging after 10^4 iterations, exiting loop');
        sqrerror = sqrerrornew;
        laststep = 0;
        lastbstep = 0;
        iterations = 0;
        layer = layer-1;
        if layer==0
            break
        end
    end
end
end




function [cost,derrordW,derrordb] = finderror(Net,inputs,desiredOut,layer)
NetOut = forward(Net,inputs);
cost = (desiredOut-NetOut).^2.*0.5;%all errors

%need to find the error in the layer inputs and outputs
layerIn = inputs;
layerOut = inputs;
for i=1:1:layer
    layerIns = layerOut;
    layerOut = forward1step(Net,layerIns,i);
end

% find the layer desired output and then the layer error
out_desired = desiredOut;
for i = 1:1:Net.nhiddenLayers+1-layer
    out_desired = -(ones(length(out_desired(:,1)),1)*Net.blayer{end+1-i}+log((1-out_desired)./out_desired))/Net.Wlayer{end+1-i};
end
error = (out_desired - layerOut);

%use cross error to prevent learning slowdown with sigmoid functions
derrordW = -2*layerIn'*(error.*(layerOut.*(1-layerOut))*Net.nodeconst)/length(out_desired(:,1));
derrordb = -2*sum(error.*(layerOut.*(1-layerOut))*Net.nodeconst,1)/length(out_desired(:,1));

%cost = sum(error.^2,2).*(0.5/length(desiredOut(:,1))) + (Net.lambda/(2*length(desiredOut(1,:))))*sum(Net.Wlayer{layer}.^2)'*ones(1,length(desiredOut(:,1)));% normalized(1/2error^2) + 1/2penalty for complex model
end 
