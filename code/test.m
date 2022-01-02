clear variables

%% Hyperparams
nRun = 1;               % used to save different runs of the same test on different files
epochs = 50;            % number of epochs
eta = [0.5 1.02];       % RPROP gains
layerSetup = [15];      % layers and feature maps
filterDim = [3 3];      % dimension of the filters
trainDim = 100;         % number of samples in training set
valDim = 20;            % number of samples in validation set
testDim = 20;           % number of samples in test set


%% DATASET
X = loadMNISTImages('mnist/train-images-idx3-ubyte');
Labels = loadMNISTLabels('mnist/train-labels-idx1-ubyte');
T = getTargetsFromLabels(Labels);

% Training set
X_train = X(:, 1 : trainDim)';
T_train = T(:, 1 : trainDim)';

% Validation set
X_val = X(:, trainDim+1 : trainDim+valDim)';
T_val = T(:, trainDim+1 : trainDim+valDim)';

% Test set
X_test = X(:, trainDim+valDim+1 : trainDim+valDim+testDim)';
T_test = T(:, trainDim+valDim+1 : trainDim+valDim+testDim)';


disp(['Epochs: ', num2str(epochs)]);
disp(['Layer setup: ', num2str(layerSetup)]);
disp(['Filter dim: ', num2str(filterDim)]);
disp(['Training set: ', num2str(trainDim), '; validation set: ', ...
    num2str(valDim), '; test set: ', num2str(testDim)]);
    
%% Create network
net = ConvNetwork([28 28 1], 10, layerSetup, filterDim);

% Learning
disp('Starting learning...');
[terr, verr, bestNet] = net.learn(X_train, T_train, X_val, T_val, epochs, eta);
%[terr, verr, bestNet] = net.learn(X_train, T_train, X_val, T_val, epochs, eta, -1, 0, 0, 'stopEpochs', 5);
figure(), grid, hold on
plot(terr), plot(verr)
legend('Training', 'Validation')

% Calculate accuracy
exact = 0;
for i = 1 : testDim
    y = bestNet.forward(X_test(i,:));
    [m, classification] = max(y);
    targetClass = find(T_test(i,:));
    if classification == targetClass
        exact = exact + 1;
    end
end
accuracy = exact / testDim;
disp(['exact: ', num2str(exact), '; total: ', num2str(testDim), '; accuracy: ', num2str(accuracy)]);

% Save on file
save([num2str(length(layerSetup)), 'layers_', num2str(layerSetup(1)), 'map_filterDim', num2str(filterDim(1)), 'x', num2str(filterDim(2))]);%, '_LONG(', num2str(nRun), ')']);