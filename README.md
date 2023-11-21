### Organizers edit
Original submission repository -> [https://github.com/ABS-Lab/EPiC_2023](https://github.com/ABS-Lab/EPiC_2023)


# EPiC_2023

The goal of the EPiC challenge was to create an open
competition to predict affective experiences from physiology
data. More details about the challenge can be found here: https://github.com/Emognition/EPiC-2023-competition/blob/main/README.md

A summary of the physiology data employed is available in the md file "physio_variables.md", in this GitHub branch.
The Python libraries and packages are available in the md file "Dependencies.md".

### FINAL RESULTS
Our final results are available in the folder *"Epic23_Northeastern_submission"*.

--------

### Our awesome Team
Our team was composed by three graduate students from the Affective and Brain Sciences Lab at Northeastern University.
Alessia Iancarelli, Kieran McVeigh, and Yiyu Wang. 

### Overview
In this work, we employ various regression models to predict
valence and arousal levels based on the down-sampled
physiological and annotation data. For each model, we
conduct scenario-dependent validation, utilizing the 1 Hz
resampled physiological and annotation data for training in
each folder.
Upon training the models and generating predictions, it was
crucial to up-sample these predictions back to 20 Hz to align
with the original testing data structure. This process enables
us to deliver the predictions of valence and arousal levels in
the 20 Hz format, meeting the study's target resolution.

### Preprocessing
You can find the scripts used for data cleaning, features extraction, and manipulation in the folder *"Preprocessing.zip".*
Data preprocessing was performed using the NeuroKit2
Python library. For each of the signals, we followed the
standard NeuroKit2 processing pipeline (see *"Physio_Clean_NK2.ipynb"*).
In addition to the physiological features, we also
conditionally included subject number, video number,
subject mean valence, subject mean arousal, video mean
arousal, and video mean valence depending on the scenario,
i.e., in scenario 2 we included video information but not
subject information, in scenario 3 vice versa.
In one version of the processing, we zscored the data within each training data. We kept both the raw and the zscored versions, and compared whether zscoring affected the performance later (see *"FeatureExtraction.ipynb"*).

### Temporal feature engineering 
We computed dynamic-related features for a subset of our physiological signals. 
For each of these signals, we calculated 5-second lagged
values, 5-second future values, 10s second rolling averages,
and 10-second rolling standard deviations, were calculated to
enrich our dataset and potentially enhance our analysis. By
incorporating these derived features, we aimed to capture
relevant temporal dynamics and variability in the
psychophysiological signals, providing a more
comprehensive representation of the underlying processes (see *"PhysioFeatureManipulation.ipynb"*).

Finally, we also included an approach that looked at
30-second windows of physiology, based on the observation
that the valence and arousal scores were relatively stable over
time so might see greater signal and mean changes over long
periods of time. (see *"30_sec_script"* folder)


### Machine learning models
*Train_test_pipeline folder* 
We explore several regression models, including Elastic Net,
Random Forest Regressor, xgboost, and Support Vector Regression
with a Radial Basis Function (RBF) kernel. For each model,
we iterate through various combinations of input features and
assess their performance using the RMSE metric (see *"All_Model_Prediction.ipynb"*).

### Validation
*Train_test_pipeline folder*
We developed three different validation procedures to
split the training data to compare different training models.
The three validation procedures matched the cross-validation
in the four scenarios. Specifically, for scenario 1, we first
calculated the ratio between the training video length and the
testing video length from the train and test dataset. We then
used the ratio to split the time points in videos in the training
set into training and validation sets. Corresponding to the
second scenario, we validated our model by splitting the
participants in the training in a 5-fold cross-validation.
Corresponding to the third and fourth scenarios, we split the
videos in the training set in a 4-fold and 2-fold cross-
validation. (see *"pipeline_helper.py"*)

### Results Summary
The final submited models: SubmissionResults.ipynb
For easy visualization: Results_Summarizer.ipynb
