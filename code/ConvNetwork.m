%%
% This class represents a convolutive network.
classdef ConvNetwork < handle
    
    % Name of the net (string). This does not have any effect on the
    % network behaviour: you can use it as a network ID.
    properties
        name
    end
    
    properties (SetAccess = private)
        layers           % Convolutive layers
        outLayer         % Output layer
        errorFuncName    % (string) Name of the error function. Inputs as before.
        inputDim         % Input dimension
    end
    
    properties (Access = private)
        errorFunc        % Pointer to error function. It has to accepts as
                         % inputs: the outputs y and the targets t
        errorDerFunc     % Pointer to derivative error function. Inputs as before.
        errorDerFuncName % (string) Name of the derivative error function. Input as before.
        needSoftmax      % 1 if softmax post-processing is needed, 0 else
    end
    
    
    methods
        
        %%
        %{
        Constructor. Inputs:
         - inputDim: dimensions of the input (e.g. [28 28 3]);
         - outputDim: dimension of the output of the net (scalar)
         - layerSetup: the i-th element of this array is the number of
                       features map to create in the i-th convolutive
                       layer. The map will have a number of convolutive
                       layers equals to the length of this array.
         - filterDim: dimensions of the filter, equals for all the layer
                      (e.g. [3 3])
        %}
        function this = ConvNetwork(inputDim, outputDim, layerSetup, filterDim)
            this.inputDim = inputDim;
            layerNum = length(layerSetup);
            this.layers = cell(1,layerNum);
            prevOutDim = inputDim;
            
            for i = 1 : layerNum
                nodesNum = ConvLayer.numOfNodes(prevOutDim, filterDim);
                
                % Check if there are too many layers (the input will
                % vanish)
                if prevOutDim(1) < filterDim(1) || prevOutDim(2) < filterDim(2)
                    disp('ERROR: layers or filter dimensions not correct');
                    return;
                end
                
                % Find the smaller max-pooling window dimension that
                % exactly subdivide the matrix
                mpDim = 0;
                for k = 2 : floor(min(nodesNum)/2)
                    if mod(nodesNum(1),k) == 0 && mod(nodesNum(2),k) == 0
                        mpDim = [k k];
                        break
                    end
                end

                % If not found, use 1x1 windows (i.e. don't apply max)
                if isscalar(mpDim)
                    mpDim = [1 1];
                end
                
                % Calculate the new output size
                this.layers{i} = ConvLayer(prevOutDim, filterDim, layerSetup(i), mpDim);
                prevOutDim = this.layers{i}.outputDim;
            end
            
            
            % Create the output layer
            this.outLayer = Layer(randn(outputDim, prod(prevOutDim)), randn(outputDim, 1), "identity", "identity");
            
            this.setErrorFunction("crossEntropy");
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
        Compute the forward propagation, given the input.
        Returns the output y of the network, and the output z (as a cell
        array) of each convolutive layer of the network.
        The last output, maxIndex, is a cell array which elements, one
        for every layer, are the maxIndex output of the ConvLayer.forward
        function.
        %}
        function [y,z,maxIndex] = forward(this, input)
            layerInput = input;
            
            % Convolutive layers
            z = cell(size(this.layers));
            maxIndex = cell(size(this.layers));
            for i = 1 : length(this.layers)
                [lout, lz, mx] = this.layers{i}.compute(layerInput);
                z{i} = {lout, lz};
                maxIndex{i} = mx;
                layerInput = lout;
            end
            
            % Output layer
            for j = 1 : size(layerInput,3)
                layerInput(:,:,j) = layerInput(:,:,j)';
            end
            layerInput = reshape(layerInput, 1, numel(layerInput));
            y = this.outLayer.compute(layerInput);
            z{length(this.layers)+1} = y;
            y = y';
            
            % Softmax
            if(this.needSoftmax)
                y = softmax(y);
            end
            for i = 1 : length(y)
                if y(i) == 0
                    y(i) = 1e-300;
                end
            end
        end
        
        %%
        % Reset the network
        function this = reset(this)
            for i = 1 : length(this.layers)
                for m = 1 : length(this.layers{i}.featureMaps)
                    tempW = this.layers{i}.featureMaps{m}.W';
                    tempD = this.layers{i}.featureMaps{m}.D';
                    
                    nonzero = find(tempW);
                    w = randn(1, length(nonzero)/size(this.layers{i}.featureMaps{m}.W,1));
                    d = .1+.1*randn(1, length(nonzero)/size(this.layers{i}.featureMaps{m}.D,1));
                    b = randn();
                    db = .1+.1*randn();
                    
                    p = 1;
                    for k = 1 : length(nonzero)
                        tempW(nonzero(k)) = w(p);
                        tempD(nonzero(k)) = d(p);
                        p = p + 1;
                        if p > length(w)
                            p = 1;
                        end
                    end
                    
                    this.layers{i}.featureMaps{m}.W = tempW';
                    this.layers{i}.featureMaps{m}.D = tempD';
                    this.layers{i}.featureMaps{m}.bias = b * ones(size(this.layers{i}.featureMaps{m}.bias));
                    this.layers{i}.featureMaps{m}.Db = db  * ones(size(this.layers{i}.featureMaps{m}.bias));
                    
                end
            end
        end
        
        %%
        % Create a new net that is a copy of this.
        function newNet = copy(this) 
            outputDim = size(this.outLayer.W,1);
            filterDim = this.layers{1}.filterDim;
            
            layerNum = length(this.layers);
            layerSetup = zeros(1, layerNum);
            for l = 1 : layerNum
                layerSetup(l) = length(this.layers{l}.featureMaps);
            end
            
            newNet = ConvNetwork(this.layers{1}.inputDim, outputDim, layerSetup, filterDim);
            newNet.setErrorFunction(this.errorFuncName, this.errorDerFuncName, this.needSoftmax);

            for l = 1 : layerNum
                newNet.layers{l} = this.layers{l};
            end
            newNet.outLayer = this.outLayer;
        end
        
        %%
        % Copy net parameters into this.
        function this = copyFrom(this, net)
            this.setErrorFunction(net.errorFuncName, net.errorDerFuncName, net.needSoftmax);
            this.layers = cell(size(net.layers));
            for l = 1 : length(this.layers)
                this.layers{l} = net.layers{l};
            end
            this.outLayer = net.outLayer;
        end
        
        %%
        % Copy this network into net.
        function copyTo(this, net)
            net.copyFrom(this);
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
                      array [eta- eta+] for RProp. Default values:
                      [0.5 1.2].
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
            if nargin < 11, earlyStop = "";     end
            if nargin < 10,  mu = 0;            end
                if mu > 1,      mu = 1;
                elseif mu < 0,  mu = 0;         end
            if nargin < 9,  wd = 0;             end
            if nargin < 8,  R = -1;             end
            
            if nargin < 7,  eta = [0.5 1.2];
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
            weightVar = cell(1,length(this.layers)+1);
            biasVar = cell(1,length(this.layers)+1);
            weightVarPrev = cell(size(weightVar));
            biasVarPrev = cell(size(biasVar));

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
                    Ed = cell(numLayer+1,1);
                    Edb = cell(numLayer+1,1);
                    for l = 1 : numLayer
                        numMap = length(this.layers{l}.featureMaps);
                        Ed{l} = cell(numMap,1);
                        Edb{l} = cell(numMap,1);
                        for m = 1 : numMap
                            Ed{l}{m} = zeros(1, prod(this.layers{l}.filterDim)*this.layers{l}.inputDim(3));
                            Edb{l}{m} = 0;
                        end
                    end
                    Ed{numLayer+1} = zeros(size(this.outLayer.W));
                    Edb{numLayer+1} = zeros(size(this.outLayer.bias));
                    
                    % Calc error derivatives
                    for n = 1 : size(xx{b},1)
                        [Edn,Edbn] = this.backProp(xx{b}(n,:), tt{b}(n,:));
                        for l = 1 : numLayer
                            for m = 1 : length(this.layers{l}.featureMaps)
                                Ed{l}{m} = Ed{l}{m} + Edn{l}{m};
                                Edb{l}{m} = Edb{l}{m} + Edbn{l}{m};
                                if wd ~= 0
                                    Ed{l}{m} = Ed{l}{m} + wd * nonzeros(this.layers{l}.featureMaps{m}.W(1,:));
                                    Edb{l}{m} = Edb{l}{m} + wd * this.layers{l}.featureMaps{m}.bias(1);
                                end
                            end
                        end
                        Ed{numLayer+1} = Ed{numLayer+1} + Edn{numLayer+1};
                        Edb{numLayer+1} = Edb{numLayer+1} + Edbn{numLayer+1};
                    end
                    
                    % Update D (RProp) ...
                    if useRProp && nep > 1
                        for l = 1 : numLayer    
                            % ... for weights
                            for m = 1 : length(this.layers{l}.featureMaps)
                                d = zeros(size(Ed{l}{m}));
                                for i = 1 : size(Ed{l}{m},2)
                                    eprod = Ed_prev{l}{m}(i) * Ed{l}{m}(i);
                                    if eprod < -RProp_TRESH
                                        d(i) = eta(1);
                                    elseif eprod > RProp_TRESH
                                        d(i) = eta(2);
                                    else
                                        d(i) = 1;
                                    end
                                end
                                    
                                p = 1;
                                for r = 1 : size(this.layers{l}.featureMaps{m}.W,1)
                                    for c = 1 : size(this.layers{l}.featureMaps{m}.W,2)
                                        if this.layers{l}.featureMaps{m}.W(r,c) ~= 0
                                            this.layers{l}.featureMaps{m}.D(r,c) = this.layers{l}.featureMaps{m}.D(r,c) * d(p);
                                            p = p+1;
                                            if p > length(d)
                                                p = 1;
                                            end
                                        end
                                    end
                                end
                                
                                    
                                if abs(this.layers{l}.featureMaps{m}.D(1,i)) > 50
                                    this.layers{l}.featureMaps{m}.D(:,i) = 50 * sign(this.layers{l}.featureMaps{m}.D(:,i));
                                end
                            
                            % ... and for biases
                                for i = 1 : numel(this.layers{l}.featureMaps{m}.Db)
                                    eprod = Edb_prev{l}{m} * Edb{l}{m};
                                    if eprod < -RProp_TRESH
                                        this.layers{l}.featureMaps{m}.Db(i) = this.layers{l}.featureMaps{m}.Db(i) * eta(1);
                                    elseif eprod > RProp_TRESH
                                        this.layers{l}.featureMaps{m}.Db(i) = this.layers{l}.featureMaps{m}.Db(i) * eta(2);
                                    end
                                end
                            end
                        end
                        
                        % Output layer
                        % ... for weights
                        for i = 1 : numel(this.outLayer.D)
                            eprod = Ed_prev{numLayer+1}(i) * Ed{numLayer+1}(i);
                            if eprod < -RProp_TRESH
                                this.outLayer.D(i) = this.outLayer.D(i) * eta(1);
                            elseif eprod > RProp_TRESH
                                this.outLayer.D(i) = this.outLayer.D(i) * eta(2);
                            end
                            if abs(this.outLayer.D(i)) > 50
                                this.outLayer.D(i) = 50 * sign(this.outLayer.D(i));
                            end
                        end
                            
                        % ... and for biases
                        for i = 1 : numel(this.outLayer.Db)
                            eprod = Edb_prev{numLayer+1}(i) * Edb{numLayer+1}(i);
                            if eprod < -RProp_TRESH
                                this.outLayer.Db(i) = this.outLayer.Db(i) * eta(1);
                            elseif eprod > RProp_TRESH
                                this.outLayer.Db(i) = this.outLayer.Db(i) * eta(2);
                            end
                        end
                        
                    end  % layers cycle
                    
                    % Save derivative for RProp
                    if useRProp
                        Ed_prev = Ed;
                        Edb_prev = Edb;
                    end
                        
                end  % nep cycle
                
                % Update weights (gradient descent with momentum)
                for l = 1 : numLayer
                    for m = 1 : length(this.layers{l}.featureMaps)
                        if useRProp
                            weightVar{l}{m} = this.layers{l}.featureMaps{m}.D;
                            biasVar{l}{m} = zeros(size(this.layers{l}.featureMaps{m}.Db));
                            for r = 1 : size(this.layers{l}.featureMaps{m}.D,1)
                                p = 1;
                                for c = 1 : size(this.layers{l}.featureMaps{m}.D,2)
                                    if ( this.layers{l}.featureMaps{m}.W(r,c) ~= 0)
                                        if (Ed{l}{m}(p) ~= 0)
                                            weightVar{l}{m}(r,c) = -sign(Ed{l}{m}(p)) .* this.layers{l}.featureMaps{m}.D(r,c);
                                        end
                                        p = p + 1;
                                    end
                                    if Edb{l}{m} ~= 0
                                        biasVar{l}{m}(r) = -sign(Edb{l}{m}) .* this.layers{l}.featureMaps{m}.Db(r);
                                    end
                                end
                            end
                        else
                            wVar = - eta * Ed{l}{m};
                            bVar = - eta * Edb{l}{m};
                            weightVar{l}{m} = zeros(size(this.layers{l}.featureMaps{m}.W));
                            biasVar{l}{m} = zeros(size(this.layers{l}.featureMaps{m}.bias));
                            for r = 1 : size(this.layers{l}.featureMaps{m}.W,1)
                                p = 1;
                                for c = 1 : size(this.layers{l}.featureMaps{m}.W,2)
                                    if ( this.layers{l}.featureMaps{m}.W(r,c) ~= 0)
                                        weightVar{l}{m}(r,c) = wVar(p);
                                        if nep > 1
                                            weightVar{l}{m}(r,c) = weightVar{l}{m}(r,c) + mu * weightVarPrev{l}{m}(r,c);
                                        end
                                        p = p + 1;
                                    end
                                    biasVar{l}{m}(r) = bVar;
                                    if nep > 1
                                        biasVar{l}{m}(r) = biasVar{l}{m}(r) + mu * biasVarPrev{l}{m}(r);
                                    end
                                end
                            end
                            
                            weightVarPrev{l}{m} = weightVar{l}{m};
                            biasVarPrev{l}{m} = biasVar{l}{m};
                            
                        end
                        
                        this.layers{l}.featureMaps{m}.W = this.layers{l}.featureMaps{m}.W + weightVar{l}{m};
                        this.layers{l}.featureMaps{m}.bias = this.layers{l}.featureMaps{m}.bias + biasVar{l}{m};
                    end
                end
                
                % Output layer
                if useRProp
                    weightVar{numLayer+1}= zeros(size(this.outLayer.D));
                    for k = 1 : numel(this.outLayer.D)
                        if Ed{numLayer+1}(k) ~= 0
                            weightVar{numLayer+1}(k) = -sign(Ed{numLayer+1}(k)) * this.outLayer.D(k);
                        end
                    end
                    biasVar{numLayer+1}= zeros(size(this.outLayer.Db));
                    for k = 1 : numel(this.outLayer.Db)
                        if Edb{numLayer+1}(k) ~= 0
                            biasVar{numLayer+1}(k) = -sign(Edb{numLayer+1}(k)) .* this.outLayer.Db(k);
                        end
                    end
                else
                    weightVar{numLayer+1} = - eta * Ed{numLayer+1};
                    biasVar{numLayer+1} = - eta * Edb{numLayer+1};
                    if nep > 1
                        weightVar{numLayer+1} = weightVar{numLayer+1} + mu * weightVarPrev{numLayer+1};
                        biasVar{numLayer+1} = biasVar{numLayer+1} + mu * biasVarPrev{numLayer+1};
                    end
                    weightVarPrev{numLayer+1} = weightVar{numLayer+1};
                    biasVarPrev{numLayer+1} = biasVar{numLayer+1};
                end
                    
                this.outLayer.W = this.outLayer.W + weightVar{numLayer+1};
                this.outLayer.bias = this.outLayer.bias + biasVar{numLayer+1};
                
                
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
        
        
        %%
        %{
        Calculate delta needed for back propagation.
        Accepts in input the output of the network, the targets of the
        dataset, the output z of every neuron in every layer, and the
        indexes of the maximum values (i.e. the last output of
        ConvNetwork.forward)
        %}
        function delta = calcDelta(this, y, t, z, max)
            numLayer = length(this.layers);
            delta = cell(numLayer+1,1);
            
            % Output layer
            delta{numLayer+1} = this.outLayer.fDerOut(z{numLayer+1}) .* this.errorDerFunc(y, t);
            
            % Convolutive layers
            for l = numLayer : -1 : 1
                for m = 1 : length(this.layers{l}.featureMaps)
                    outDerFunc = this.layers{l}.featureMaps{m}.fDerOut;
                    
                    % Calcolo del delta del Max - pooling
                    if l == numLayer
                        numConn = numel(z{l}{1}(:,:,m));
                        firstIndex = (m-1)*numConn + 1;
                        endIndex = m*numConn;
                        delta{l}{2}{m} = outDerFunc(z{l}{1}(:,:,m)) .* ((this.outLayer.W(:,firstIndex:endIndex))' * delta{l+1}');
                    else
                        for k = 1 : size(this.layers{l}.outputDim, 3)
                            numConn = numel(z{l}{1}(:,:,k));
                            firstIndex = (k-1)*numConn + 1;
                            endIndex = k*numConn;
                            for mm = 1 : length(this.layers{l+1}.featureMaps)
                                temp = (this.layers{l+1}.featureMaps{mm}.W(:,firstIndex:endIndex))' * delta{l+1}{1}{k};
                                if exist('prodDelta', 'var') == 1
                                    prodDelta = prodDelta + temp;
                                else
                                    prodDelta = temp;
                                end
                            end
                        end
                        delta{l}{2}{m} = outDerFunc(z{l}{1}(:,:,m)) .* prodDelta;
                        clear prodDelta;
                    end
                    delta{l}{2}{m} = reshape(delta{l}{2}{m}', [numel(delta{l}{2}{m}), 1]);
                    
                    
                    % Feature - maps
                    % delta {LAYER} {1 = feature map, 2 = max-pooling}
                    maxDelta = delta{l}{2}{m}(max{l}{m});
                    maxDelta = reshape(maxDelta', [numel(maxDelta), 1]);
                    delta{l}{1}{m} = outDerFunc(z{l}{2}{m}) .* (ones((this.layers{l}.outputDim(1)*this.layers{l}.outputDim(2)), prod(this.layers{l}.nodesSize))' * maxDelta);
                    delta{l}{1}{m} = reshape(delta{l}{1}{m}', [numel(delta{l}{1}{m}), 1]);
                    
                end
            end
        end
        
        %%
        % Calculate derivative of errors and biases using back propagation.
        % Accepts in input the dataset input and target
        function [Ed,Edb] = backProp(this, x, t)
            [y,z,m] = this.forward(x);
            delta = this.calcDelta(y, t, z, m);
            
            if ~isvector(x)
                for j = 1 : size(x,3)
                    x(:,:,j) = x(:,:,j)';
                end
                x = reshape(x, [1, numel(x)]);
            end

            % Calculate derivatives
            numLayer = length(this.layers);
            Ed = cell(numLayer+1,1);
            Edb = cell(numLayer+1,1);
            
            for l = 1 : numLayer
                mapNum = length(this.layers{l}.featureMaps);
                Ed{l} = cell(mapNum,1);
                Edb{l} = cell(mapNum,1);
                for m = 1 : mapNum
                    % Feature maps weights
                    if l == 1
                        Ed_temp = delta{l}{1}{m} * x; % x is a row
                    else
                        zz = z{l-1}{1};
                        for i = 1 : size(zz,3)
                            zz(:,:,i) = zz(:,:,i)';
                        end
                        zz = reshape(zz, [1, numel(zz)]);
                        Ed_temp = delta{l}{1}{m} * zz;
                    end
                    
                    Ed_temp = reshape(Ed_temp', 1, numel(Ed_temp));
                    W_temp = reshape(this.layers{l}.featureMaps{m}.W', 1, numel(this.layers{l}.featureMaps{m}.W));
                    p = 1;
                    for k = 1 : numel(Ed_temp)
                        if(W_temp(k) ~= 0)
                            Ed{l}{m}(p) = Ed_temp(k);
                            p = p+1;
                        end
                    end
                    Ed{l}{m} = reshape(Ed{l}{m}', prod(this.layers{l}.filterDim)*this.layers{l}.inputDim(3), size(this.layers{l}.featureMaps{m}.W,1))';
                    Ed{l}{m} = sum(Ed{l}{m},1);
                    
                    % Feature maps biases
                    Edb{l}{m} = sum(delta{l}{1}{m}, 'all');
                    
                end
            end
            
            
            % Output layer
            outL = numLayer+1;
            Ed{outL} = zeros(size(this.outLayer.W));
            Edb{outL} = zeros(size(this.outLayer.bias));
            
            lastOut = z{numLayer}{1};
            for j = 1 : size(lastOut,3)
                lastOut(:,:,j) = lastOut(:,:,j)';
            end
            lastOut = reshape(lastOut, 1, numel(lastOut));
            
            Ed{outL} = delta{outL}' * lastOut;
                
            if(isrow(delta{outL}))
                Edb{outL} = delta{outL}';
            else
                Edb{outL} = delta{outL};
            end
        end
        
    end
    
end