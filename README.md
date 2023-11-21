### Organizers edit
Original submission repository -> [https://github.uconn.edu/lrm22005/Cafeteros-EPiC-2023-competition](https://github.uconn.edu/lrm22005/Cafeteros-EPiC-2023-competition)


# EPiC 2023: The Emotion Physiology and Experience Collaboration - TEAM "CAFETEROS"

## Team members
Hugo Posada-Quintero (1), Fernando Marmolejo-Ramos (2), Javier Pinzón-Arenas (3), Luis Mercado-Díaz (1), Carlos Barrera-Causil (4), Jorge Ivan Padilla (4), Raydonal Ospina (5), and Julian Tejada(6)

(1) University of Connecticut, (2) University of South Australia, (3) Universidad Militar Nueva Granada, (4) Instituto Tecnologico Metropolitano, (5) Federal University of Bahia , and Universidade Federal de Sergipe (6)

## Approach explanation

For this competition, we proposed the use of a hybrid architecture containing a temporal convolutional neural network (TCN) and a stacked bi and uni-directional long short-term memory network (SBU-LSTM) arranged in parallel. This is intended to give to the model the capability of learning the spatial features of the signals with the TCN, and the temporal dynamics with the LSTM.

## Content
This repository contains the preprocessing of the signals, data structuring, and training/testing of the models.

### Preprocessing:
Five time series were computed from the available signals: instantaneous heart rate (IHR) from ECG, pulse transit time (PTT) from ECG and BVP, tonic component of GSR --also called EDA-- (EDAton), phasic component of EDA (EDAph), y respiratory rate (RESP). All the signals were comuted at 20 Hz in agreement with emotions labeling. 

For IHR, The Pah-Tompkins algorithm was used for detecting R peaks from the ECG signal, IHR computed and resampled at 20 Hz using spline. 
For PTT, BVP peaks were detected using BP annotate, and the time between each R peak and its subsequent BVP peak was computed. The time series was even sampled at 20 Hz using spline. 

For EDA decomposition into EDAton and EDAph, cvxEDA algorithm was used. The EDA signal was previously downsampled to 20 Hz. 

RESP was computed using peaks detected from the respiration signal, the respiratory rate signal obtained, and resampled at 20 Hz using spline. 

Run "PreProcessingData.m" to create csv files with the time series explained above.

### Data Structuring:
Once the data has been processed, running "data_struct.m", the .csv files of each scenario/fold will be stored in a .mat file in ".\data\" folder, as a struct variable named "dataframe" with the fields "test" and "train".

### Training and Testing the models
For training, it is only necessary to run the "Training_Networks.m", and it will use the previously structured dataframes. For each scenario/folder, 2 architectures are trained: one for Valence and other for Arousal, using 90% of the data for training and 10% for validation. Once the training has finished, the two models of each scenario/fold are stored in ".\Models\" folder, as a struct variable named "model_data". The struct contains the "Arousal" and "Valence" fields with the model ("Net"), training information ("Info"), and the hyperparameters used in the models ("Hyperparameters").

For testing, it is used the "Testing_Networks.m", and it will run the models with the test dataset, saving the results of each scenario in ".\results_mat\" folder.

Use "ExportAttempt.m" to export the results from .mat files to .csv files ready for submission or RMSE-based evaluation.

This repository contains data, information, and helpful code for the [EPiC 2023 competition](https://epic-collab.github.io/competition/).



