function [x_train, y_train_valence, y_train_arousal, x_val, y_val_valence, y_val_arousal] = Setup_data(dataframe)
%
% Setup_data create the database cells necessary to train the models for the EPiC-2023-competition
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

train_subj = fieldnames(dataframe.train);       % subjects in the dataframe
num_data = length(train_subj);                  % number of videos
num_val = floor(num_data / 10);                 % number of video samples in the validation fold (10% of the total)

signals = {};
valence = {};
arousal = {};


% build the dataset with each of the signals
% signals: (IHR, PTT, Phasic EDA, Tonic EDA, Resp)
for subj = 1 : num_data
    signals{subj,1} = table2array(dataframe.train.(train_subj{subj})(:,2:6))';
    valence{subj,1} = table2array(dataframe.train.(train_subj{subj})(:,end-1))';
    arousal{subj,1} = table2array(dataframe.train.(train_subj{subj})(:,end))';
end

% Training set
x_train = signals(num_val+1:end, 1);
y_train_valence = valence(num_val+1:end, 1);
y_train_arousal = arousal(num_val+1:end, 1);

% Validation set
x_val = signals(1:num_val, 1);
y_val_valence = valence(1:num_val, 1);
y_val_arousal = arousal(1:num_val, 1);


disp("Building training and validation datasets complete")

end