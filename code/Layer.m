%{
This class rapresents a neural network layer, it is characterized
by a Weight matrix [number of nodes, number of inputs] a Bias vector
[number of nodes] the Activation and Output functions of the neurons of the
layer. This class also stores variables used in the learning process:
D, Db, fDerOutName and fDerOut;
%}

classdef Layer
    
    properties
        W;              % 2-d numeric array
        D;              % update distance (RPROP)
        Db;             % update distance (RPROP) for biases
        bias;           % 1-d numeric array
        fAttName        % string  
        fAtt;           % functiom
        fOutName;       % string
        fOut;           % function
        fDerOutName;    % string
        fDerOut;        % function
    end
    
    methods
        
        %{
        CONSTRUCTOR. Inputs:
         - w: weight matrix of the layer with dimentions [number of nodes, number of inputs]
         - b: bias vector with dimension equal to the number of nodes
         - fa: name of the activation function
         - fo: name of theoutput function
        %}
        function this = Layer(w, b, fa, fo)
            
            this.W = w;
            this.D = (.1 + .1*randn(size(this.W))) .* (w~=0);
            
            this.bias = b;
            this.Db = .1+.1*randn(size(this.bias));
            this.fAttName = fa;
            this.fOutName = fo;
            this.fDerOutName = "der"+fo;
            
            % Pre-defined relu and identity functions, if another function
            % is needed. Appropriate files must be created as has already
            % been done with the sigmoid (see sigmoid.m and devsigmoid.m)
           
            if(this.fOutName == "identity")
                this.fOut = @(x) x;
                this.fDerOut = @(x) 1;
            elseif (this.fOutName == "relu")
                this.fOut = @(x) max(0,x);
                this.fDerOut = @(x) x>0;
            else
                this.fOut = str2func(this.fOutName);
                this.fDerOut = str2func(this.fDerOutName);
            end
            if(this.fAttName == "identity")
                this.fAtt = @(x) x;
            elseif (this.fAttName == "relu")
                this.fAtt = @(x) max(0,x);
            else
                this.fAtt = str2func(this.fAttName);
            end
                
        end
        
        %{
        compute. Computes the outputs of the layer.
        Inputs:
         - this: a layer object
         - input: an input vector of adequate size for the layer
        Outputs:
         - output: an output vector of the same size of the numer of nodes of
        the layer
        %}
        function output=compute(this, input)
            if (size(input,1) == 1)
                input = input';
            end
            
            a = (this.W * input) + this.bias;
            
            for i = 1 : length(a)
              a(i) = this.fAtt(a(i));
            end           
            for i = 1 : length(a)
              a(i) = this.fOut(a(i));
            end
            output = a;
        end
        
    end
    
end
            