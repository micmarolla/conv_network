%{
This class represents a convolutive layer, that is a certain number of
features maps plus a max-pooling for each of them.
Each feature map layer is represented as a Layer object; of course, those
layers are completely indipendent from each other.
Note that max-pooling is not achieved as a different layer, but as an
operation directly applied to the feature maps output.
%}
classdef ConvLayer
    
    properties
        featureMaps     % Cell array of layer
        nodesSize       % Dimension of nodes matrix
        filterDim       % Dimension of filters
        MPWinDim        % Dimension of max-pooling windows
        inputDim        % Dimension of the input
        outputDim       % Dimension of the output
    end
    
    methods
        
        %%
        %{
        Constructor. Inputs:
         - inputDim: dimension of the input matrix, e.g. [28 28 3]
         - filterDim: dimension of the filter, e.g. [3 3]. Both dimensions
                       must be odd, so that the filter has a center pixel.
         - mapNum: number of feature maps in the convolutive layer
         - MPWinDim: dimension of max-pooling windows, e.g. [2 2]
        %}
        function this = ConvLayer(inputDim, filterDim, mapNum, MPWinDim)
            
            % Check that filter dimensions are odd 
            if mod(filterDim(1),2) == 0  || mod(filterDim(2),2) == 0
                disp('Filters must have odd dimensions');
                return
            end
            
            this.filterDim = filterDim;
            
            if length(inputDim) < 3
                inputDim(3) = 1;
            end
            
            this.inputDim = inputDim;
            this.MPWinDim = MPWinDim;
            
            % Computes the number of nodes
            this.nodesSize = this.numOfNodes(inputDim, filterDim);
            nodesNum = prod(this.nodesSize);
            
            this.featureMaps = cell(1,mapNum);
            % For each feature map:
            for m = 1 : mapNum
                % Generate random weights and biases. This are stored in
                % temporary variables, so that each node will be
                % initialized with the same set of weights and biases.
                tempWeight = randn(1,prod(filterDim)*inputDim(3));
                tempD = randn(1,prod(filterDim)*inputDim(3));
                tempBias = randn();
                tempDb = randn();
                
                % Initializes weights and biases of the feature map
                W = zeros(nodesNum, inputDim(1)*inputDim(2));
                D = zeros(nodesNum, inputDim(1)*inputDim(2));
                B = tempBias * ones(nodesNum,1);
                Db = tempDb * ones(1,nodesNum);
                
                %% Populate the weight matrix W
                
                % Center of the FIRST row of the filter
                center = floor(filterDim(2)/2);
                
                % For each node (i.e. each pixel, excluding the outer ones),
                % find every pixel that is inside the filter centered in
                % that pixel.
                for i = 1 : nodesNum
                    for k = 1 : filterDim(1)  % For each row of the filter
                        
                        % If the filter will be split, do not consider it
                        % and jump to the next one
                        if mod(center,inputDim(1)) >= inputDim(2)-floor(filterDim(2)/2)
                           nextRow = ceil(center/inputDim(2));
                           center = nextRow*inputDim(2) + 1;
                        end
                        
                        firstIndex = center-floor(filterDim(2)/2) + (k-1)*inputDim(2) + 1;
                        lastIndex = center+floor(filterDim(2)/2) + (k-1)*inputDim(2) + 1;
                        
                        W(i, firstIndex : lastIndex) = 1;
                    end
                    
                    center = center + 1;
                    
                end
                
                
                % Create the layer
                colsW = size(W,2);
                extendedW = zeros(size(W,1), colsW*inputDim(3));
                for i = 1 : inputDim(3)
                    extendedW(:,(i-1)*colsW+1 : i*colsW) = W;
                end
                
                colsD = size(D,2);
                extendedD = zeros(size(D,1), colsD*inputDim(3));
                for i = 1 : inputDim(3)
                    extendedD(:,(i-1)*colsD+1 : i*colsD) = D;
                end
                
                
                % Put the random generated weights on the previously
                % found connections
                for i = 1 : nodesNum
                    p = 1;
                    for j = 1 : size(extendedW,2)
                        if extendedW(i,j) == 1
                            extendedW(i,j) = tempWeight(p);
                            extendedD(i,j) = tempD(p);
                            p = p+1;
                        end
                    end
                end
                
                
                this.featureMaps{m} = Layer(extendedW,B,"relu","identity");
                this.featureMaps{m}.D = extendedD;
                this.featureMaps{m}.Db = tempDb;
                
            end
            
            % Calculate output dimension
            this.outputDim = zeros(1,3);
            this.outputDim(1) = this.nodesSize(1)/this.MPWinDim(1);
            this.outputDim(2) = this.nodesSize(2)/this.MPWinDim(2);
            this.outputDim(3) = length(this.featureMaps);
            
        end
        
        
        %%
        %{
        Compute the output of the layer, given the input as a matrix.
        Returns both the output (3D matrix) of the layer, and the output
        of every feature map (as cell array). The last output, maxIndex,
        is a cell array of matrix (one for each feature map): the i,j
        element of the matrix is the index of the maximum input of the
        i,j node.
        %}
        function [out, fm, maxIndex] = compute(this,input)
            % If input is matrix, converts it as a row
            if ~isvector(input)
                for j = 1 : size(input,3)
                    input(:,:,j) = input(:,:,j)';
                end
                input = reshape(input, 1, numel(input));
            end
            
            % Initializes output array
            out = zeros(this.outputDim);
            fm = cell(size(this.featureMaps));
            
            maxIndex = cell(size(this.featureMaps));
            
            for m = 1 : length(this.featureMaps)
                fm{m} = this.featureMaps{m}.compute(input);
                
                % Reshape the feature map output as a matrix
                MP = reshape(fm{m}', [this.nodesSize(1), this.nodesSize(2)]);
                
                % Subdivide this matrix in windows
                MPmat = mat2cell(MP, this.MPWinDim(1)*ones(1,this.outputDim(1)), this.MPWinDim(2)*ones(1,this.outputDim(2)));
                
                % Apply max-pooling
                for i = 1 : this.outputDim(1)
                    for j = 1 : this.outputDim(2)               
                        [out(i,j,m), maxIndex{m}(i,j)] = max(MPmat{i,j}, [], 'all', 'linear');
                    end
                end
                
            end
        end
    end
    
    methods (Static)
        
        %%
        %{
        Calculate the number of nodes in the feature maps, given the
        dimensions of the input and of the filters.
        %}
        function nodesSize = numOfNodes(inputDim, filterDim)
            nodesSize = zeros(1,2);
            nodesSize(1) = inputDim(1) - 2*floor(filterDim(1)/2);
            nodesSize(2) = inputDim(2) - 2*floor(filterDim(2)/2);
        end
    end
    
end