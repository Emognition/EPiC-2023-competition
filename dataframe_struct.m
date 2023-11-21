%% dataframe constructor
clear
disp('CSV files to .mat dataframe')

data_path{1} = {'.\data\scenario_1\'};

data_path{2} = {'.\data\scenario_2\fold_0\';
               '.\data\scenario_2\fold_1\';
               '.\data\scenario_2\fold_2\';
               '.\data\scenario_2\fold_3\';
               '.\data\scenario_2\fold_4\'};

data_path{3} = {'.\data\scenario_3\fold_0\';
               '.\data\scenario_3\fold_1\';
               '.\data\scenario_3\fold_2\';
               '.\data\scenario_3\fold_3\'};

data_path{4} = {'.\data\scenario_4\fold_0\';
               '.\data\scenario_4\fold_1\'};

% Loop for each scenario
for scenario = 1 : length(data_path)

    % Loop for each fold inside the scenarios
    for sce_fold = 1:length(data_path{scenario})
        clear dataframe
        
        % Train
        datastruct = dir([data_path{scenario}{sce_fold}, 'train\preprocessed\*.csv']);
        N_subjects = numel(datastruct);    % number of data


        for n_sub = 1 : N_subjects

            clear table_data
            disp(datastruct(n_sub).name(1:end-4))

            table_data = readtable([datastruct(n_sub).folder, '/', datastruct(n_sub).name]);
            dataframe.train.(datastruct(n_sub).name(1:end-4)) = table_data;

        end



        % Test
        datastruct = dir([data_path{scenario}{sce_fold}, 'test\preprocessed\*.csv']);
        N_subjects = numel(datastruct);    % number of data


        for n_sub = 1 : N_subjects

            clear table_data
            disp(datastruct(n_sub).name(1:end-4))

            table_data = readtable([datastruct(n_sub).folder, '/', datastruct(n_sub).name]);
            dataframe.test.(datastruct(n_sub).name(1:end-4)) = table_data;

        end


        % Save Data
        if scenario == 1
            filename = '.\data\scenario_1.mat';
        elseif scenario == 2
            filename = ['.\data\scenario_2_fold_', num2str(sce_fold-1), '.mat'];
        elseif scenario == 3
            filename = ['.\data\scenario_3_fold_', num2str(sce_fold-1), '.mat'];
        else
            filename = ['.\data\scenario_4_fold_', num2str(sce_fold-1), '.mat'];
        end
        save(char(filename),'dataframe');

        disp(' ')
        disp('Dataset has been saved!')
    end

end



