### Organizers edit
Original submission repository -> [https://github.com/AffectiveBulls/EPiC-Challenge-Results](https://github.com/AffectiveBulls/EPiC-Challenge-Results)

# EPiC-Challenge-Results

## Team: Saurabh Hinduja<sup>:one:</sup>, Maneesh Bilalpur<sup>:one:</sup>, Liza Jivnani<sup>:two:</sup>, Shaun Canavan<sup>:two:</sup>
### <sup>:one:</sup> University of Pittsburgh, <sup>:two:</sup> University of South Florida


# Method:
Our work is inspired by [Hinduja et al. 2021](https://ieeexplore.ieee.org/abstract/document/9597422) 's variance fusion approach where multiple physiological signals have been weighted by their variance and combined into a single signal for prediction. We used variance fusion to generate a time-series that is representative of all the physiological signals available through the challenge. This time-series was then split into windows of length = 50 (since annotation is available once per every 50 samples) to train a MultiLayer Feed Forward neural network that predicts valence and arousal. The network was optimized for MSE loss.

These instructions below help in setting up the required environment and how to use the code.

# Environment

To install necessary packages, use `pip install -r ./Code/requirements.txt`

Once successfully installed, use the following instructions to organize the data and run different scenarios.

# Organizing the data

Create a new folder on the same level as `Code` and `results` by name `data`.

The data organization follows the challenge convention. Place challenge data such that Scenario-2 data is organised as 

`./data/scenario-2/fold-0/test/annotations` for annotations and `./data/scenario-2/fold-0/test/physiology` for corresponding physiological data

# Running Scenarios

## Training for scenarios

Go to `Code` folder.

Training code for each scenario starts with `Training` and its corresponding testing code starts with `Testing`

To train for scenario-1, from `Code` folder run

`python Training_Scenario_1.py`

Similarly for scenario-2, from `Code` folder run

`python Training_Scenario_2.py`

Scenario-3, from `Code` folder run

`python Training_Scenario_3_4.py`

Scenario-4, from `Code` folder run

`python Training_Scenario_3_4.py`


## Testing for scenarios

To test for scenario-1, from `Code` folder run

`python Testing_Scenario_1.py`

Similarly for scenario-2, from `Code` folder run

`python Testing_Scenario_2.py`

Scenario-3, from `Code` folder run

`python Testing_Scenario_3_4.py`

Scenario-4, from `Code` folder run

`python Testing_Scenario_3_4.py`


Submitted test predictions can be found in `./results/scenario_X/` organized as per the requirements of the Challenge. If testing is run, these predictions would be overwritten.