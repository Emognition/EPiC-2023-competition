clear;
close all;
clc;

addpath(".\utils\")

% Train Final Model for each Scenario

data_path{1} = {'.\Models\scenario_1\'};

data_path{2} = {'.\Models\scenario_2\fold_0\';
               '.\Models\scenario_2\fold_1\';
               '.\Models\scenario_2\fold_2\';
               '.\Models\scenario_2\fold_3\';
               '.\Models\scenario_2\fold_4\'};

data_path{3} = {'.\Models\scenario_3\fold_0\';
               '.\Models\scenario_3\fold_1\';
               '.\Models\scenario_3\fold_2\';
               '.\Models\scenario_3\fold_3\'};

data_path{4} = {'.\Models\scenario_4\fold_0\';
               '.\Models\scenario_4\fold_1\'};


% Models hyperparameters in each scenario given as:
    % Network_hyperparameters.(...) = [scenario_1; scenario_2; scenario_3; scenario_4];

% For arousal
Network_hyperparameters.Arousal.hidden_units =  [200; 150; 150; 150];
Network_hyperparameters.Arousal.num_blocks =    [8; 4; 4; 4];
Network_hyperparameters.Arousal.num_filters =   [64; 64; 64; 64];
Network_hyperparameters.Arousal.filter_size =   [5; 5; 5; 5];

% For valence
Network_hyperparameters.Valence.hidden_units =  [100; 200; 150; 200];
Network_hyperparameters.Valence.num_blocks =    [8; 4; 8; 4];
Network_hyperparameters.Valence.num_filters =   [64; 128; 64; 128];
Network_hyperparameters.Valence.filter_size =   [5; 3; 5; 3];

% Mini-batch size at each iteration
batch_size = 8;            


%% Training loop

% Loop for each scenario
for scenario = 1 : length(data_path)

    % Loop for each fold inside the scenarios
    for sce_fold = 1: length(data_path{scenario})
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Load dataframe
        if scenario == 1
            name_sce = 'scenario_1';
            load('.\data\scenario_1.mat')
        elseif scenario == 2
            name_sce = ['scenario_2_fold_', num2str(sce_fold-1)];
            load(['.\data\scenario_2_fold_', num2str(sce_fold-1), '.mat'])
        elseif scenario == 3
            name_sce = ['scenario_3_fold_', num2str(sce_fold-1)];
            load(['.\data\scenario_3_fold_', num2str(sce_fold-1), '.mat'])
        else
            name_sce = ['scenario_4_fold_', num2str(sce_fold-1)];
            load(['.\data\scenario_4_fold_', num2str(sce_fold-1), '.mat'])
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        %% Arousal
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        clear model_data

        % Set the hyperparameter of the model for each scenario/fold
        n_blocks =  Network_hyperparameters.Arousal.num_blocks(scenario);
        n_filts =   Network_hyperparameters.Arousal.num_filters(scenario);
        filt_size = Network_hyperparameters.Arousal.filter_size(scenario);
        n_hidden =  Network_hyperparameters.Arousal.hidden_units(scenario);


        %%%%% Train Network Arousal %%%%%%
        fprintf("==================================== \n")
        fprintf(" \nScenario %i", scenario)
        fprintf(" \nFold %i", (sce_fold-1))
        fprintf(" \nTrain Arousal\n")
        fprintf("==================================== \n")

        % Setup Dataset
        [x_train, ~, y_train_arousal, ...
            x_val, ~, y_val_arousal] = Setup_data(dataframe);

        % Setup Architecture
        num_features = height(x_train{1});
        layers = TCN_LSTM_architecture(n_hidden, filt_size, n_filts, num_features, n_blocks);

        % Define the training options
        train_size = length(x_train);
        opts = trainopt({x_val, y_val_arousal}, batch_size, train_size);

        % Training
        [net, info] = trainNetwork(x_train, y_train_arousal, layers, opts);

        model_data.Arousal.Net = net;
        model_data.Arousal.Info = info;

        % Store the parameters used
        model_data.Arousal.Hyperparameters.Hidden_Units = n_hidden;
        model_data.Arousal.Hyperparameters.Num_Blocks = n_blocks;
        model_data.Arousal.Hyperparameters.Num_Filters = n_filts;
        model_data.Arousal.Hyperparameters.Filter_Size = filt_size;

        fprintf("Arousal Training Finished \n \n")



        %% Valence
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        % Set the hyperparameter of the model for each scenario/fold
        n_blocks =  Network_hyperparameters.Valence.num_blocks(scenario);
        n_filts =   Network_hyperparameters.Valence.num_filters(scenario);
        filt_size = Network_hyperparameters.Valence.filter_size(scenario);
        n_hidden =  Network_hyperparameters.Valence.hidden_units(scenario);


        %%%%% Train Network Valence %%%%%%
        fprintf("==================================== \n")
        fprintf(" \nScenario %i", scenario)
        fprintf(" \nFold %i", (sce_fold-1))
        fprintf(" \nTrain Valence\n")
        fprintf("==================================== \n")

        % Setup Dataset
        [x_train, y_train_valence, ~, ...
            x_val, y_val_valence, ~] = Setup_data(dataframe);

        % Setup Architecture
        num_features = height(x_train{1});
        layers = TCN_LSTM_architecture(n_hidden, filt_size, n_filts, num_features, n_blocks);

        % Define the training options
        train_size = length(x_train);
        opts = trainopt({x_val, y_val_valence}, batch_size, train_size);

        % Training
        [net, info] = trainNetwork(x_train, y_train_valence, layers, opts);

        model_data.Valence.Net = net;
        model_data.Valence.Info = info;

        % Store the parameters used
        model_data.Valence.Hyperparameters.Hidden_Units = n_hidden;
        model_data.Valence.Hyperparameters.Num_Blocks = n_blocks;
        model_data.Valence.Hyperparameters.Num_Filters = n_filts;
        model_data.Valence.Hyperparameters.Filter_Size = filt_size;

        fprintf("Valence Training Finished \n \n")

        % Save the Network Information Structure
        net_name = 'Parallel_TCN_SBU_LSTM';
        save([data_path{scenario}{sce_fold}, 'ModelData_', name_sce, '_', net_name, '.mat'], 'model_data');

        
    end
end







%% training options function
function opts = trainopt(ValidationDS, batch_size, train_size)

opts = trainingOptions('adam', ...
    'InitialLearnRate', 1e-3, ...
    'LearnRateSchedule', 'piecewise', ...
    'Shuffle', 'every-epoch', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 25, ...
    'MaxEpochs', 100, ...
    'ExecutionEnvironment', 'auto', ...
    'miniBatchSize', batch_size,...
    'Verbose', true, ...
    'VerboseFrequency', floor(train_size / batch_size), ...
    'ValidationData',ValidationDS, ...
    'ValidationFrequency',floor(train_size / batch_size), ...
    'OutputNetwork','best-validation-loss');

end