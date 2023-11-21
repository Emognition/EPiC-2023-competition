### Organizers edit
Original submission repository -> [https://github.com/claire-whiting/EPiC](https://github.com/claire-whiting/EPiC)


# EPiC-challenge Submission

## Team Members: 
(listed in alphabetical order)  

- Isabel Berwian  
- Jamie Chiu
- Dan Mircea
- Erik Nook
- Henna Vartiainen
- Claire Whiting 

We are a team of researchers from the Department of Psychology and the Princeton Neuroscience Institute at Princeton University. Our core team comprises four PhD students, Jamie Chiu, Dan Mircea, Henna Vartiainen, and Claire Whiting, who work in Niv Lab and Logic of Emotion Lab.  

Additionally, we have two advisory team members: Isabel Berwian, a postdoctoral researcher from Niv Lab, and Erik Nook, the director of Logic of Emotion Lab.

## Approach:

For each scenario, we used a slightly adapted version of an LSTM model. We used windows of 100ms, either prior or centered, to predict each valence and arousal rating. For scenario 1 we also used one-hot encodings for subjects and videos as input features, to account for the fact that the prediction is within sample (within subject/video pair). To build and refine our models and hyperparameters, we divided the training dataset into a partial training set and a validation set, with the exact split depending on the scenario. For instance, in Scenario 1, the last 15s (300 annotations) of each sample were held out for validation. In Scenario 2, six subjects were used as validation in each fold. We padded the training samples with 0s to account for the their varying lengths. Following this, we trained the best model for each scenario using the full training dataset of that scenario, which is what we submitted in the notebooks. Further specifics on the approach used for each scenario can be found in each scenario's notebook.

We uploaded the notebook used for each scenario. Each notebook begins with loading and pre-processing the data. A model is trained and saved per fold -- the models are also uploaded in the repository and loaded into the notebook at the end to generate the predicted values of valence and arousal. The predictions are saved into the csv files in the results directory, as per instructions provided. The original naming and structure of directories are kept, e.g., `./results/scenario_2/fold_3/test/annotations/sub_0_vid_2.csv`.

---

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
