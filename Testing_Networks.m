clear;
close all;
clc;

addpath(".\utils\")

% Test Final Model for each Scenario

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

net_name = 'Parallel_TCN_SBU_LSTM';


%% Test loop

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
        dataframe = rmfield(dataframe,"train");

        % Load model_data
        load([data_path{scenario}{sce_fold}, 'ModelData_', name_sce, '_', net_name, '.mat'])

        % Setup Dataset
        [x_test, y_test_valence, y_test_arousal] = Setup_data_Test(dataframe);


        fprintf("\n====================================\n")
        fprintf(" \nScenario %i", scenario)
        fprintf(" \nFold %i", (sce_fold-1))
        
        %% Arousal
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% Test Network Arousal %%%%%%
        fprintf("\n====================================\n")
        fprintf("Test Arousal")
        fprintf("\n====================================\n")

        % Setup model
        net_arousal = model_data.Arousal.Net(1);

        % Testing
        predictions_arousal = predict(net_arousal,x_test, 'MiniBatchSize', 8);




        %% Valence
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%% Test Network Valence %%%%%%
        fprintf("====================================\n")
        fprintf("Test Valence")
        fprintf("\n====================================\n")

        % Setup model
        net_valence = model_data.Valence.Net(1);

        % Testing
        predictions_valence = predict(net_valence,x_test, 'MiniBatchSize', 8);




        %% Save predictions in test dataframe
        test_subj = fieldnames(dataframe.test);       % subjects in the dataframe
        num_data = length(test_subj);  
        
        for subj = 1 : num_data
            dataframe.test.(test_subj{subj})(:,end-1:end) = table(predictions_valence{subj}',predictions_arousal{subj}');
        end

        save(['.\results_mat\scenario_', num2str(scenario), '\test_', name_sce, '.mat'], 'dataframe');

        
    end
end
