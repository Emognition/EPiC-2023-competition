function [x_test, y_test_valence, y_test_arousal] = Setup_data_Test(dataframe)
%
% Setup_data_Test create the database cells to be entered to the predict() matlab function
%
% Input:
%   dataframe       = struct containing the test data of each sub_vid
%
% Output:
%   x_test          = cell containing the signals arrays of each sub_vid
%                     (IHR, PTT, Phasic EDA, Tonic EDA, Resp)
%   y_test_valence  = cell containing the arrays of valence values of each sub_vid
%   y_test_arousal  = cell containing the arrays of arousal values of each sub_vid



disp("Building Data")

test_subj = fieldnames(dataframe.test);       % subjects in the dataframe
num_data = length(test_subj);                 % number of videos

signals = {};
valence = {};
arousal = {};


% build the dataset with each of the signals
% signals: (IHR, PTT, Phasic EDA, Tonic EDA, Resp)
for subj = 1 : num_data
    signals{subj,1} = table2array(dataframe.test.(test_subj{subj})(:,2:6))';
    valence{subj,1} = table2array(dataframe.test.(test_subj{subj})(:,end-1))';
    arousal{subj,1} = table2array(dataframe.test.(test_subj{subj})(:,end))';
end


x_test = signals(:, 1);
y_test_valence = valence(:, 1);
y_test_arousal = arousal(:, 1);


disp("Building test dataset complete")

end