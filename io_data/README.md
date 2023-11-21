# Input and Output data

![input data](./data.png)

> Generate the input and output data from clean data and features.

Using one input data to predict one arousal and one valence.
It consists of past 50 clean signals (50 * 7 dim), current clean signal (7 dim), future 50 clean signal (50 * 7 dim), and 69 features at that time step.

- Input: 776 dim
- Output: 2 dim

## training data

- scenario 1: one input corresponds to one subject and one video (240 input data)
- scenario 2: one input corresponds to one video and multiple subjects (8 input data per fold, 40 input data in total)
- scenario 3
  - two input data for training arousal model: video 0, 3, 4, 21; video 16, 20, 10, 22
  - two input data for training valence model: video 10, 22, 4, 21; video 0, 3, 16, 20
- scenario 4: one input corresponds to one video and multiple subjects (4 input data per fold, 8 input data in total)

## test data

One test data corresponds to one subject and one video.
