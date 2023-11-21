function layers = TCN_LSTM_architecture(numHiddenUnits, filterSize, numFilters, numFeatures, numBlocks)
%
% TCN_LSTM_architecture : constructor of the Parallel TCN-SBU-LSTN architecture
%
% Input:
%   numHiddenUnits  = Number of hidden units for the SBU-LSTN section
%   filterSize      = Filter size of the 1-dimention convolutional filter
%   numFilters      = Number of filters used in the convolutional layer of the TCN sectio
%   numFeatures     = Number of inputs (signals) of the architecture
%   numBlocks       = Number of blocks to be used in the TCN section
%
% Output:
%   layers          = lgraph of the architecture


% TCN layers
tcnLayers = TCN_Arq(filterSize, numFilters, numBlocks, numFeatures);

% LSTM layers
lstmLayers = LSTM_Arq(numHiddenUnits);

%% Parallel CNN-LSTM

% Branch 1 (TCN)
% Concatenation Input + TCN activation features
layer = concatenationLayer(1,2,'Name','concat_1');
layers = addLayers(tcnLayers,layer);
layers = connectLayers(layers,tcnLayers.Layers(end).Name,'concat_1/in1');

% Output of the network
outlayers = [ ...
    fullyConnectedLayer(1,'Name','fc')
    regressionLayer('name','Regression')
    ];

% Addition of the output to the TCN layers
layers = addLayers(layers,outlayers);
% Concatenation
layers = connectLayers(layers,'concat_1','fc');

% Branch 2 (LSTM)
layers1 = lstmLayers;

% Addition of the LSTM layers to the TCN
layers = addLayers(layers,layers1);
% Connection of the LSTM to the input 
layers = connectLayers(layers,'input',layers1(1).Name);
% Concatenation (TCN + LSTM features)
layers = connectLayers(layers,layers1(2).Name,'concat_1/in2');

end





%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function lstmlayers = LSTM_Arq(numHiddenUnits)
% Stacked Bi- and Uni-directional lstm architecture
lstmlayers = [ ...
    bilstmLayer(numHiddenUnits,'OutputMode','sequence', 'Name', 'bilstm1')
    lstmLayer(numHiddenUnits,'OutputMode','sequence', 'Name', 'lstm2')
    ];
end




%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function layers = TCN_Arq(filterSize, numFilters, numBlocks, numFeatures)

%% TCN
% Spatial dropout factor of the TCN block
dropoutFactor = 0.005;

layer = sequenceInputLayer(numFeatures,Name="input");
lgraph = layerGraph(layer);

outputName = layer.Name;

for i = 1:numBlocks
    % Dilation factor of the causal 1-d conv layer for each block
    dilationFactor = 2^(i-1);

    % TCN block
    layers = [
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal",Name="conv1_"+i)
        layerNormalizationLayer
        spatialDropoutLayer(dropoutFactor)
        convolution1dLayer(filterSize,numFilters,DilationFactor=dilationFactor,Padding="causal")
        layerNormalizationLayer
        reluLayer
        spatialDropoutLayer(dropoutFactor)
        additionLayer(2,Name="add_"+i)];

    % Add and connect layers
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph,outputName,"conv1_"+i);

    % Skip connection
    if i == 1
        % Include convolution in first skip connection
        layer = convolution1dLayer(1,numFilters,Name="convSkip");

        lgraph = addLayers(lgraph,layer);
        lgraph = connectLayers(lgraph,outputName,"convSkip");
        lgraph = connectLayers(lgraph,"convSkip","add_" + i + "/in2");
    else
        lgraph = connectLayers(lgraph,outputName,"add_" + i + "/in2");
    end

    % Update layer output name
    outputName = "add_" + i;
end

layers = lgraph;
end