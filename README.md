### Organizers edit
Original submission repository -> [https://github.com/klean2050/epic_sail](https://github.com/klean2050/epic_sail)


<div align="center">

# EPiC 2023 Challenge - USC SAIL submission
This repo contains code implementation as well as trained models for our ACII challenge submission.
  
</div>

## Installation

We recommend using a conda environment with ``Python >= 3.9`` :
```
conda create -n epic_sail python=3.9
conda activate epic_sail
```
Clone the repository and install the dependencies:
```
git clone https://github.com/klean2050/epic_sail
pip install -e epic_sail
```
You will also need the ``ecg-augmentations`` library:
```
git clone https://github.com/klean2050/ecg-augmentations
pip install -e ecg-augmentations
```

## Project Structure

```
tiles_ecg_model/
├── setup.py             # package installation script
├── config/              # configuration files for each train session
├── results/             # results folder (same structure as in dataset)
└── src/                 # main project directory
    ├── loaders/             # pytorch dataset classes
    ├── models/              # backbone neural network models
    ├── scripts/             # preprocess, training and evaluation
    ├── trainers/            # lightning classes for each train session
    └── utils/               # miscellaneous scripts and methods
```

## TILES Dataset

Tracking Individual Performance with Sensors (TILES) is a project holding multimodal data sets for the analysis of stress, task performance, behavior, and other factors pertaining to professionals engaged in a high-stress workplace environments. Biological, environmental, and contextual data was collected from hospital nurses, staff, and medical residents both in the workplace and at home over time. Labels of human experience were collected using a variety of psychologically validated questionnaires sampled on a daily basis at different times during the day. In this work, we utilize the TILES ECG data from [here](https://tiles-data.isi.edu/), to pre-train an ECG encoder to assist the affect recognition process.

## Pre-Training Framework

Each TILES participant has their ECG recorded for 15 seconds every 5 minutes during their work hours, for a total of 10 weeks. To preprocess TILES ECG data, navigate to the root directory and run the following command:
```
python src/scripts/preprocess.py
```

We pre-train the model in a self-supervised manner, through applying and then predicting different transformations overlaid onto the ECG signals. To augment the ECG samples we use the [PyTorch ECG Augmentations](https://github.com/klean2050/ecg-augmentations) package. A lightweight encoder is used to extract latent representations from the augmented data inputs, which is based on a 6-layer state-space model called S4. We use a light architecture (4 MB in size) to make the model applicable to real-time settings. By recognizing the applied transformations, the model is forced to identify the underlying association between augmented versions of the same sample. Pre-training is done with the following command:
```
python src/scripts/ssl_pretrain.py
```
We already provide a trained checkpoint at ``ckpt`` so you do not need to run this process.

## Fine-Tuning Framework

We transfer the trained ECG encoder to the downstream tasks in a late fusion setting, where additional sensor streams are trained from scratch at similar architectures for the valence/arousal estimation task, along with aligning their latent representations to those produced by the (non-frozen) ECG model. HThe final state estimation is done by concatenating the embeddings of the different modalities and then applying an attention layer before the MLP classifier. Apart from ECG, modalities that are used include: SCR extracted from EDA at 20 Hz and clean RSP signal at 20 Hz. One can configure the parameters involved in the respective config file, and then run:
```
python src/scripts/supervised_epic.py
```

## Submission

We fine-tune a model instance for every scenario and fold and save the test predictions at ``.npy`` files. We randomly select a subset of training data as validation set based on each scenario's assumptions and based on the performance on this set we select to submit the predictions of either the pre-trained ECG encoder, the pre-trained fusion network, or trained from scratch. We use the code in ``make_submission`` to construct the ``results`` folder.

## Results & Checkpoints

To view results in TensorBoard run (after initiating training):
```
tensorboard --logdir ./runs
```

## Acknowledgements

In this study we made use of the following repositories:

* [VCMR](https://github.com/klean2050/VCMR)


## Authors
* [Kleanthis Avramidis](https://klean2050.github.io): PhD Student in Computer Science, USC SAIL
