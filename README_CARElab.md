# introducing the team

Our team is formed by two information science master students, and three professors at different levels. Our specialties are on machine learning, signal processing, and affective research. We are a multinational team, our institutions are located in Japan and The Netherlands. We come from five different countries. Finally, we are keen to understand better the relationship between bodily changes and subjective experience.

## Xin Wei

## Huakun Liu

## Felix Dollack

## Chirag Raman

## Kiyoshi Kiyokawa

## Hideaki Uchiyama

## Monica Perusquia-Hernandez


# explaining your approach

We used a theoretical assumption approach to train multiple weak classifiers, together with late fusion. This approach was done independently for each scenario. General signal preprocessing was done in advance. As an output layer in the algorithm pipeline, we applied a low-pass filter to remove sudden annotation variations that are unlikely to happen.

# describing the repository content
The repository is organized in folders as follows:
- **CARElab** : exploratory analysis notebooks
- **data** : raw data provided for the challenge
- **features** : data cleaning and feature extraction
- **io_data** : input and output data generation
- **models** : model training
- **results** : solution data files for the challenge submission
- **src** : util scripts

# how to run the code
Use Python 3.10.6

## Challenge
To reproduce the submitted challenge results:
1. `features/features.py` -> for generating clean signals and features
2. `io_data/io_data.py` -> for generating the specific input for each scenario
3. `models/train_models.py` -> for training the models
4. `results/test_models.py` -> for predicting the results

> Please enter the directory of each script to execute it.
> 
> See each script for detailed usage description.

Example (Scenario 1)
```sh
# generate clean signals and features (-s scenario)
$ cd features
$ python3 features.py -s 1 
# generate specific input
# Note -- you might encounter the following error:
#  FileExistsError: [Errno 17] File exists...
# To resolve the problem simply rerun the same command.
$ cd ../io_data
$ python3 io_data.py -s 1 -t 'train'
$ python3 io_data.py -s 1 -t 'test'
# train the model (-n num_gpus)
$ cd ../models
$ python3 train_models.py -s 1 -n 1
# predict the result for test data
$ cd ../results
$ python3 test_models -s 1
```

## Delay analysis
For reporduction of the data used in the delay plots execute the following code in order:
```
cd CARElab
python3 train_LSTM_lag_models.py
python3 test_LSTM_lag_models.py
```
This wil produce CSV files in the form of `performance_XXX.csv`, where XXX is either a short for the physiological signal used (e.g. ecg) or `all` if all signals were used as input.

Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

## Troubleshooting
If you see an error similar to `libcublas.so.11: undefined symbol: cublasLtGetStatusString` run:
```
pip uninstall nvidia_cublas_cu11
```
