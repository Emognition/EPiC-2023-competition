%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exports results from mat files.

clear all
close all
Fs = 20;
addpath(".\utils\")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
net_name = 'Parallel_TCN_SBU_LSTM';
max_fold = [0 4 3 1];

for n_attempt = 1 : 3
    attemptstr = attemptstrings{n_attempt};
    disp(attemptstr)
    for n_scenario = 1 : 4
        disp(['scenario ',num2str(n_scenario)])
        for n_sce_fold = 0 : max_fold(n_scenario)
            disp(['Fold ',num2str(n_sce_fold)])
            if n_scenario == 1 
                load([pwd,'\results_mats\',attemptstr,'\scenario_',num2str(n_scenario),'\test_scenario_',num2str(n_scenario),'.mat']);
                pathstr = ['\scenario_',num2str(n_scenario),'\test\'];   
            else
                load([pwd,'\results_mats\',attemptstr,'\scenario_',num2str(n_scenario),'\test_scenario_',num2str(n_scenario),'_fold_',num2str(n_sce_fold),'.mat']);
                pathstr = ['\scenario_',num2str(n_scenario),'\fold_',num2str(n_sce_fold),'\test\'];   
            end
            
            
            test_subj = fieldnames(dataframe.test);       % subjects in the dataframe
            num_data = length(test_subj);  
            
            for subj = 1 : num_data
                TableAnnotations = readtable([pwd,'/data',pathstr,'annotations\',test_subj{subj},'.csv']);
                if isequal(TableAnnotations.time,dataframe.test.(test_subj{subj}).time(201:801))
                    disp(['Time vectors agree. Saving ',test_subj{subj},'...'])
                    TableAnnotations.valence = dataframe.test.(test_subj{subj}).Valence(201:801);
                    TableAnnotations.arousal = dataframe.test.(test_subj{subj}).Arousal(201:801);
                else
                    disp('Warning: The time vectors do not coincide')
                    keyboard
                end
                writetable(TableAnnotations,[pwd,'/results',pathstr,'annotations\',test_subj{subj},'.csv'])
            end
        end
    end
end

