%%
%{
This class represents a feed-forward multi-layer neural network.
The network object is characterized by its name and layers, for each
layer number of nodes, activation function and error function can be
specified. It is also possible to make the network not full connected.
%}

classdef Network < handle

    % Name of the net (string). This does not have any effect on the
    % network behaviour: you can use it as a network ID.
    properties
        name
    end
    
    properties (SetAccess = private)
        layers           % Cell array of layers
        errorFuncName    % (string) Name of the error function. Inputs as before.
    end
    
    properties (Access = private)
        errorFunc        % Pointer to error function. It has to accepts as
                         % inputs: the outputs y and the targets t
        errorDerFunc     % Pointer to derivative error function. Inputs as before.
        errorDerFuncName % (string) Name of the derivative error function. Input as before.
        needSoftmax      % 1 if softmax post-processing is needed, 0 else
    end
    

    methods
    
    
        %{
        Constructor. Inputs:
         - Xdim: Number of input of the network (integer value). 
         - L: Array containing the number of nodes of the layers (ex. two
                layers with 3 nodes each: [3 3]).
         - O: array of string containing the name of the output function
                for each layer. Default is 'identity'.
         - A: array of string containing the name of the activation
                function for each layer. Default is 'sigmoid'.
         - EF: string containing the name of the error function.
                Default is 'crossEntropy'.
         - EFd: string containing the name of the error function derivate. 
                Default is 'dercrossEntropy'.
         - softmax: 1 if softmax is needed, false else
         - FC: cell array of matices used to specify missing connections in
                the network. Default value creates a full connected network. 
        %}
        function this = Network(XDim, L, O, A, EF, EFd, softmax, FC)
            for i = 1 : length(L)
                % Weights and bias
                if i == 1
                    weight = randn(L(1), XDim);
                else
                    weight = randn(L(i), L(i-1));
                end
                % Bias
                bias = randn(L(i), 1);
                
                % Output and activation function
                if nargin < 3
                    fout = "identity";
                else
                    fout = O(i);
                end
                if nargin < 4
                    fatt = "sigmoid";
                else
                    fatt = A(i);
                end
                if i == length(L)
                    fatt = "identity";
                end
                
                if nargin > 7 
                   [row, col]=find(FC{i});
                   for j=1:(length(row))
                   weight(row(j),col(j))=0;
                   end
                end
                %create the layers
                this.layers{i} = Layer(weight, bias, fatt, fout);
                %set the error function
                if nargin<6
                    this.setErrorFunction("crossEntropy");
                else
                    if nargin < 7
                        softmax = 0;
                    end
                    this.setErrorFunction(EF, EFd, softmax);
                end
            end
        end

        %{
        Compute the forward propagation of the neural network
        Inputs:
         - this: a Network object
         - input: an input vector of adequate size for the Network
        Outputs:
         - y: a cell array containing the output of each layer
         - z: the outputs of the last layer
        %}
        function [y,z] = forward(this, input)
            z = cell(size(this.layers));
            z{1}=compute(this.layers{1}, input);
            for i= 2 : length(this.layers)
               z{i}=compute(this.layers{i}, z{i-1});
            end
            y = z{length(z)}';
            if(this.needSoftmax)
                y = softmax(y);
            end
        end
        
        %{
        setConnectionMap. Removes connections in the Network
        Inputs:
         - this: a Network object
         - FC: a cell array of matrices used to specify missing connections
                in the network (0 for no connection, 1 else).
        %}
        function this = setConnectionMap(this,FC)
            for i=1:length(this.layers)
                [row, col]=find(FC{i}==0);
                for j=1:(length(row))
                    this.layers{i}.W(row(j),col(j))=0;
                end
            end
        end
        
        %{
        Randomize the values of the weights of the Network
        Inputs:
         - this: a Network object
        %}
        function this = reset(this)
            for i = 1 : length(this.layers)
                [row, col]=size(this.layers{i}.W);
                
                temp = randn(row,col);
                this.layers{i}.W = this.layers{i}.W ~= 0;
                this.layers{i}.W = this.layers{i}.W .* temp;
                
                this.layers{i}.bias = randn(row,1);
                
                this.layers{i}.D = (.1+.1*randn(size(this.layers{i}.W))) .* (this.layers{i}.W~=0);
                this.layers{i}.Db = .1+.1*randn(size(this.layers{i}.bias));
            end
        end
        
    
        %%
        % Create a new net that is a copy of this.
        function newNet = copy(this)
            numOfLayers = length(this.layers);
            numOfInputs = size(this.layers{1},2);
            nodes = zeros(1,numOfLayers);
            for i = 1 : numOfLayers
                nodes(i) = size(this.layers{i},2);
            end
            
            newNet = Network(numOfInputs, nodes);
            newNet.setErrorFunction(this.errorFuncName, this.errorDerFuncName, this.needSoftmax);
            for l = 1 : numOfLayers
                newNet.layers{l} = this.layers{l};
            end
        end
        
        %%
        % Copy net parameters into this.
        function this = copyFrom(this, net)
            this.setErrorFunction(net.errorFuncName, net.errorDerFuncName, net.needSoftmax);
            this.layers = cell(size(net.layers));
            for l = 1 : length(this.layers)
                this.layers{l} = net.layers{l};
            end
        end
        
        %%
        % Copy this network into net.
        function copyTo(this, net)
            net.copyFrom(this);
        end
    
        %%
        %{
        Set error function. You can set a pointer to a user-defined
        function, or use strings "sumOfSquares" and "crossEntropy" to use
        already-defined standard error functions.
        Set softmax param to 1 if you need this kind of post-processing,
        i.e. if the output function do not produce outputs which sum is 1
        (default value: 0).
        %}
        function this = setErrorFunction(this, func, derFunc, softmax)
            if nargin < 4,  softmax = 0;    end
            this.errorFuncName = func;
            if func == "sumOfSquares"
                this.errorFunc = @(y,t) (1/2) * sum((y-t) .^ 2);
                this.errorDerFunc = @(y,t) y-t;
                softmax = 0;
            elseif func == "crossEntropy"   % cross entropy with softmax
                this.errorFunc = @(y,t) -sum(t.*log(y));
                this.errorDerFunc = @(y,t) y-t;
                softmax = 1;
            else
                this.errorFunc = str2fun(func);
                this.errorDerFuncName = derFunc;
                this.errorDerFunc = str2fun(derFunc);
            end
            this.needSoftmax = softmax;
        end
        
        
        %%
        %{
        Learn from the dataset. Params:
          - x:        input matrix; each input has to be on different
                      rows, each feature on different columns.
          - t:        target matrix; target for different inputs has to
                      be on different rows
          - x_val:    input from validation set
          - t_val:    target from validation set
          - epochs:   max number of epochs (Default: 1000)
          - eta:      learning rate. If batch is used, you can set an
                      array [eta- eta+] for RProp. Default: 0.005.
          - R:        number of elements in mini-batches. Use -1 or a
                      value greater than the number of rows of x to use
                      batch version (Default: -1)
          - wd:       weight decay factor (Default: 0)
          - mu:       momentum factor (Default: 0)
          - earlyStop: use 'firstMin' if you want to stop training after
                       the first error minimum is encountered;
                       use 'stopAfter' to stop after a certain amount of epochs,
                       specified by the param 'stopEpochs', in which the
                       validation error do not improve.
                      (Default: empty string)
          - stopEpochs: param used with 'stopEpochs' early stopping
        Outputs are the training error, the validation error, and the
        network that performed best. Howere, this object will also
        contains the best parameter set.
        %}
        function [trainErr, valErr, bestNet] = learn(this, x, t, ...
                x_val, t_val, epochs, eta, R, wd, mu, earlyStop, stopEpochs)
            % Checks
            if isempty(this.errorFunc) || isempty(this.errorDerFunc)
                disp('Error function not defined');
                return;
            end
        
            % Default params
            if nargin < 12, stopEpochs = 1;     end
            if nargin < 11, earlyStop = "";     end
            if nargin < 10,  mu = 0;            end
                if mu > 1,      mu = 1;
                elseif mu < 0,  mu = 0;         end
            if nargin < 9,  wd = 0;             end
            if nargin < 8,  R = -1;             end
            
            if nargin < 7,  eta = 0.0005;
            elseif ~isscalar(eta)
                if (eta(1) < 0 || eta(1) > 1),  eta(1) = 0.5;     end
                if (eta(2) < 1 || eta(2) > 2),  eta(2) = 1.2;     end
                if eta(1) > eta(2)
                    [eta(2), eta(1)] = deal(eta(1), eta(2)); % swap elements
                end
            end
            
            if nargin < 6,  epochs = 1000;      end
            
            % Reset network
            this.reset();
            
            % Generate mini-batches
            isBatch = 0;    % 1 if doing batch, 0 if mini-batch
            if R > 0
                batch = R * ones(1,ceil(length(x)/R));
                if size(x,1) > R && mod(size(x,1),R) ~= 0
                    batch(length(batch)) = mod(size(x,1),R);
                elseif size(x,1) < R
                    batch(length(batch)) = size(x,1);
                    isBatch = 1;
                end
            else
                batch = size(x,1);
                isBatch = 1;
            end
            
            useRProp = 0;
            if isBatch && ~isscalar(eta)
                useRProp = 1;
                RProp_TRESH = 1e-12;
            end
            
            xx = mat2cell(x, batch, size(x,2));
            tt = mat2cell(t, batch, size(t,2));
            
            trainErr = zeros(1, epochs);
            valErr = zeros(1, epochs);
            valErrIncreasing = 0;   % Number of epochs in which validation error is not improving
            numLayer = length(this.layers);
            
            % Previous weight and bias variations (used for momentum)
            weightVar = cell(size(this.layers));
            biasVar = cell(size(this.layers));
            for l = 1 : length(this.layers)
                weightVar{l} = zeros(size(this.layers{l}.W));
                biasVar{l} = zeros(size(this.layers{l}.bias));
            end
            
            
            % Calculate the error
            initerr = 0;
            for n = 1 : size(x,1)
                 y = this.forward(x(n,:));
                 initerr = initerr + this.errorFunc(y,t(n,:));
            end
            disp(['init err: ', num2str(initerr)]);
            
            
            % Init best network
            bestNet = this.copy();
            bestErr = +inf;
            
            % Through all the epochs ...
            for nep = 1 : epochs
                randDisp = randperm(length(xx));
                xx = xx(randDisp); % Shuffle batches
                tt = tt(randDisp);
                
                % Through all mini-batches ...
                for b = 1 : length(xx)
                    % Init Ed to zero
                    Ed = cell(numLayer,1);
                    Edb = cell(numLayer,1);
                    for l = 1 : numLayer
                        Ed{l} = zeros(size(this.layers{l}.W));
                        Edb{l} = zeros(size(this.layers{l}.bias));
                    end
                    
                    % Calc error derivatives
                    for n = 1 : size(xx{b},1)
                        [Edn,Edbn] = this.backProp(xx{b}(n,:), tt{b}(n,:));
                        for l = 1 : numLayer
                            Ed{l} = Ed{l} + Edn{l} + wd * this.layers{l}.W;
                            Edb{l} = Edb{l} + Edbn{l} + wd * this.layers{l}.bias;
                        end
                    end
                    
                    
                    % Update D (RProp) ...
                    if useRProp && nep > 1
                        for l = 1 : numLayer    
                            % ... for weights
                            for i = 1 : numel(this.layers{l}.D)
                                eprod = Ed_prev{l}(i) * Ed{l}(i);
                                if eprod < -RProp_TRESH
                                    this.layers{l}.D(i) = this.layers{l}.D(i) * eta(1);
                                elseif eprod > RProp_TRESH
                                    this.layers{l}.D(i) = this.layers{l}.D(i) * eta(2);
                                end
                                if abs(this.layers{l}.D(i)) > 50
                                    this.layers{l}.D(i) = 50 * sign(this.layers{l}.D(i));
                                end
                            end
                            
                            % ... and for biases
                            for i = 1 : numel(this.layers{l}.Db)
                                eprod = Edb_prev{l}(i) * Edb{l}(i);
                                if eprod < -RProp_TRESH
                                    this.layers{l}.Db(i) = this.layers{l}.Db(i) * eta(1);
                                elseif eprod > RProp_TRESH
                                    this.layers{l}.Db(i) = this.layers{l}.Db(i) * eta(2);
                                end
                            end
                        end
                    end
                    
                    % Save derivative for RProp
                    if useRProp
                        Ed_prev = Ed;
                        Edb_prev = Edb;
                    end
                        
                end
                
                % Update weights
                for l = 1 : numLayer
                    if useRProp
                        weightVar{l}= zeros(size(this.layers{l}.D));
                        for k = 1 : numel(this.layers{l}.D)
                            if Ed{l}(k) ~= 0
                                weightVar{l}(k) = -sign(Ed{l}(k)) * this.layers{l}.D(k);
                            end
                        end
                        biasVar{l}= zeros(size(this.layers{l}.Db));
                        for k = 1 : numel(this.layers{l}.Db)
                            if Edb{l}(k) ~= 0
                                biasVar{l}(k) = -sign(Edb{l}(k)) .* this.layers{l}.Db(k);
                            end
                        end
                        
                    else  % Gradient descent with momentum
                        weightVar{l} = - eta * Ed{l} + mu * weightVar{l};
                        biasVar{l} = - eta * Edb{l} + mu * biasVar{l};
                    end
                    
                    this.layers{l}.W = this.layers{l}.W + weightVar{l};
                    this.layers{l}.bias = this.layers{l}.bias + biasVar{l};
                end
                
                
                % Calculate error on training set
                for n = 1 : size(x,1)
                     y = this.forward(x(n,:));
                     trainErr(nep) = trainErr(nep) + this.errorFunc(y,t(n,:));
                end
                trainErr(nep) = trainErr(nep);
                
                % Calculate error on validation set
                for n = 1 : size(x_val,1)
                     y = this.forward(x_val(n,:));
                     valErr(nep) = valErr(nep) + this.errorFunc(y,t_val(n,:));
                end
                valErr(nep) = valErr(nep);
                
                % Get best network
                valErrIncreasing = valErrIncreasing + 1;
                if (valErr(nep) < bestErr)
                    valErrIncreasing = 0;   % Reset counter
                    bestErr = valErr(nep);
                    bestNet = this.copy();
                end
                
                disp(['epoch: ', num2str(nep), ', train_err: ', ...
                    num2str(trainErr(nep)), ', val_err: ', num2str(valErr(nep))]);
                
                
                % Check for early stopping
                if nep > 2
                    if earlyStop == "firstMin" && valErr(nep) > valErr(nep-1)
                        return  % Skip the copy of the network
                        
                    elseif earlyStop == "stopEpochs" && valErrIncreasing > stopEpochs
                        return
                        
                    end
                end
            end
           
            % Retrieve the best param
            this.copyFrom(bestNet);
            
        end

    end
    
    
    methods (Access = private)
    
        %%
        %{
        Calculate delta needed for back propagation.
        If standard functions are used (sigmoid as activation function
        and identity as ouput function of neurons, and sum of squares or
        cross entropy with soft max as error function), delta of the
        last layer are computed in a more efficient way.
        Accepts in input the output of the network, the targets of the
        dataset, and the output z of every neuron in every layer.
        %}
        function delta = calcDelta(this, y, t, z)
            numLayer = length(this.layers); % = l
            delta = cell(numLayer,1);
            for l = numLayer : -1 : 1
                outDerFunc = this.layers{l}.fDerOut;
                if (l == numLayer)
                    if ((this.errorFuncName == "sumOfSquares" || this.errorFuncName == "crossEntropy") ...
                            && ((this.layers{l}.fAttName == "sigmoid" && this.layers{l}.fOutName == "identity") ...
                            || (this.layers{l}.fAttName == "identity" && this.layers{l}.fOutName == "sigmoid")))
                        delta{l} = (y - t)';
                    else
                        delta{l} = outDerFunc(z{l}) .* this.errorDerFunc(y, t);
                    end
                else
                    delta{l} = outDerFunc(z{l}) .* ((this.layers{l+1}.W)'* delta{l+1}'); 
                end
            end
        end
        
        
        %%
        % Calculate derivative of errors and biases using back propagation.
        % Accepts in input the dataset input and target
        function [Ed,Edb] = backProp(this, x, t)
            
            [y,z] = this.forward(x);
            delta = this.calcDelta(y, t, z);

            % Calculate derivatives
            numLayer = length(this.layers);
            Ed = cell(numLayer,1);
            Edb = cell(numLayer,1);
            
            for l = 1 : numLayer
                Ed{l} = zeros(size(this.layers{l}.W));
                Edb{l} = zeros(size(this.layers{l}.bias));
                
                if l == 1
                    Ed{l} = delta{l} * x;
                else
                    Ed{l} = delta{l}' * z{l-1}';
                end
                
                if(isrow(delta{l}))
                    Edb{l} = delta{l}';
                else
                    Edb{l} = delta{l};
                end
            end
        end
        
    end

end