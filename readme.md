### Organizers edit
Original submission repository -> [https://github.com/tomdamelio/EPiC-2023-Challenge](https://github.com/tomdamelio/EPiC-2023-Challenge)


# Emotion Physiology and Experience Collaboration (EPiC) Shared Task Repository

This repository contains the work done as part of the Emotion Physiology and Experience Collaboration (EPiC) shared task competition. The goal of the competition is to develop models and methods for understanding and predicting moment-to-moment ratings of emotion using measures of physiology.

## Authors

Tomas D'Amelio, Nicolás Bruno, Leandro Bugnon, Federico Zamberlan, and Enzo Tagliazucchi

## Repository Structure

- `./preprocessing`: This directory contains scripts for preprocessing and generating continuous features from the raw data. These scripts handle the cleaning, transformation, and preparation of the data for the subsequent modeling stage.
  
- `./test`: This directory contains the final scripts for generating predictions using the preprocessed data. These scripts load the preprocessed data, train the models, and generate predictions for the four scenarios proposed in the competition.
  
- `./results`: This directory contains the final prediction results for each of the four scenarios proposed in the competition.

## Getting Started

### Installation

First, it is necessary to create a new environment. You can do this by running the following command in your terminal:

```shell
python -m venv epic_challenge
.\epic_challenge\Scripts\activate
```

Next, install the requirements file to set up the necessary packages in your environment. You can do this by running the following command in your terminal:

```shell
pip install -r requirements.txt
```

### Data

Raw data should be stored in the `/data/raw/` directory. Inside this directory, the data should be sorted into 4 different scenarios corresponding to the problem statement. Each scenario should have its own dedicated sub-directory. The structure should look like this:

`/data/raw/scenario_1`
`/data/raw/scenario_2`
`/data/raw/scenario_3`
`/data/raw/scenario_4`


### Preprocessing

After the raw data is correctly placed, the next step is to preprocess this data. To do this, run the following script:

```shell
python preprocessing/preprocessing.py
```

This script processes the raw data for each scenario and generates a set of preprocessed files. These files are crucial for the subsequent steps of model training and prediction generation. The preprocessed files will be stored in the directory: `./data/preprocessed`

### Testing

Once the preprocessing is complete, you can move onto testing. The `./test` directory contains the scripts used for generating predictions using the preprocessed data. Each scenario has its unique script file:

- Scenario 1: `test_scenario_1.py`
- Scenario 2: `test_scenario_2.py`
- Scenario 3: `test_scenario_3.py`
- Scenario 4: `test_scenario_4.py`

**Note**: There is a folder named `/ACII_figures` that contains the scripts necessary for generating the figures used in our ACII 2023 workshop paper. Please refer to this folder if you are interested in replicating or understanding the visual representations from the publication.

### Results

The prediction results for each of the four scenarios are stored in the `./results` directory.

By following these steps, you should be able to correctly run all the scripts and reproduce the results as originally intended.

## Contact

For any additional information or queries, feel free to contact me by email: dameliotomas@gmail.com
