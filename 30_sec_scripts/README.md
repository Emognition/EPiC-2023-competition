MODELS WITH MOVING AVERAGE OF 30 SEC

In this folder, we applied a rolling window function with a window size of 30 seconds (sampling rate of 1 Hz) to compute the moving average for both the annotation data and the physiological data. The purpose of calculating the moving average was to smooth the data and minimize noise. After calculating the moving average, we resampled the data by selecting every 30th row. Next, we downsampled the annotation data to a sampling rate of 1 Hz to match the frequency of the physiological data. This process ensures that the annotation data is compatible and synchronized with the physiological data, allowing for more accurate comparisons and analyses between the two datasets.


Script order:

1) Moving_avg_30sec_window: 
compute the moving average for both the annotation data and the physiological data

2) downsample_annotations:
downsample the annotation data to a sampling rate of 1 Hz to match the frequency of the physiological data

3) models_30sec:
train several models to predict valence and arousal across scenarios

4) test_data_models_30s
make the predictions for the valence and arousal annotations.
