import neurokit2 as nk
import numpy as np
import pandas as pd
import scipy
from scipy import signal
import more_itertools
from scipy.stats import zscore


def emg_process(emg_signal, threshold=0.1, sampling_rate=1000, filterCutoff=5, savGolWinLen = 0):
    # Set default Savitsky-Golay Window length
    if savGolWinLen == 0:
        savGolWinLen = sampling_rate

    # Sanitize input
    emg_signal = nk.signal_sanitize(emg_signal)

    # Clean signal
    # emg_cleaned = emg_clean(emg_signal, sampling_rate=sampling_rate)
    emg_cleaned = emg_clean(emg_signal, filterCutoff=filterCutoff, sampling_rate=sampling_rate)

    # Windowing
    step_size = 1 #slide
    window_size = int(round(0.1*sampling_rate)) #51 #window of 100ms at 512Hz = 51 samples
    # calculate z-scores
    emg_z = zscore(emg_cleaned)
    # calculate RMS
    rms_data = []
    windowed_data = list(more_itertools.windowed(emg_z, window_size, step=step_size))
    leftover_size = len(emg_z) - len(windowed_data)
    for window in windowed_data:
        rms_data.append(np.sqrt(np.square(window).mean(axis=0)))
    rms_data.extend([0]*leftover_size)
    # low pass to further remove noise from the rms
    
    # Further smooth the EMG envelope with a Savitsky-Golay filter of third order
    if (sampling_rate%2)==0:
        # savgol window length must be odd
        savGolWinLen = sampling_rate+1
    rms_data = signal.savgol_filter(rms_data, savGolWinLen, 3)

    # Get amplitude
    amplitude = nk.emg_amplitude(emg_cleaned)

    # Get onsets, offsets, and periods of activity
    # activity_signal, info = nk.emg_activation(
    #     amplitude, sampling_rate=sampling_rate, threshold="default"
    # )

    # use threshold * stddev (instead of 0.1*stddev)
    activity_signal, info = emg_activity(amplitude, threshold=threshold, sampling_rate=sampling_rate)
    info["sampling_rate"] = sampling_rate  # Add sampling rate in dict info

    # Prepare output
    signals = pd.DataFrame(
        {"EMG_Raw": emg_signal, "EMG_Clean": emg_cleaned, "EMG_Rms": rms_data, "EMG_Amplitude": amplitude}
    )

    signals = pd.concat([signals, activity_signal], axis=1)

    return signals, info


def emg_clean(emg_signal, sampling_rate=1000, filterCutoff=5):
    emg_signal = nk.as_vector(emg_signal)

    f_notch = np.array([60, 120, 180, 240]) # notch frequencies to filter out because of line noise
    width   = np.ones(len(f_notch)) * 3;              # width/2 of each notch
    cutlow  = 5    # lower cut-off frequency for elliptic filter
    cuthigh = 250  # upper cut-off frequency for elliptic filter

    # Missing data
    n_missing = np.sum(np.isnan(emg_signal))
    if n_missing > 0:
        warn(
            "There are " + str(n_missing) + " missing data points in your signal."
            " Filling missing values by using the forward filling method.",
            category=nk.NeuroKitWarning,
        )
        emg_signal = _emg_clean_missing(emg_signal)

    # Parameters
    # order = 4
    # frequency = filterCutoff
    # frequency = (
    #     2 * np.array(frequency) / sampling_rate
    # )  # Normalize frequency to Nyquist Frequency (Fs/2).

    # filter it
    clean = local_filter(emg_signal, f_notch, width, cutlow, cuthigh, sampling_rate)

    # # Filtering
    # b, a = scipy.signal.butter(N=order, Wn=frequency, btype="highpass", analog=False)
    # filtered = scipy.signal.filtfilt(b, a, emg_signal)

    # # Baseline detrending
    # clean = nk.signal_detrend(filtered, order=0)

    return clean

# =============================================================================
# Handle missing data
# =============================================================================
def _emg_clean_missing(emg_signal):

    emg_signal = pd.DataFrame.pad(pd.Series(emg_signal))

    return emg_signal

# Functions taken from matlab implementation following Gruebler 2014
def notch_filter(data, notch, width, fs):
    fa = (notch - width) / (fs/2)
    fb = (notch + width) / (fs/2)
    Wn = np.array([fa, fb])
    order = 4
    b, a = signal.butter(order, Wn, btype='stop')
    return signal.filtfilt(b, a, data)

def elliptic_filter(data, flow, fhigh, fs):
    Wn    = np.array([flow, fhigh]) * (2/fs)
    order = 4
    maxRipple      = 0.1
    minAttenuation = 40
    b, a  = signal.ellip(order, maxRipple, minAttenuation, Wn, btype='stop')
    return signal.filtfilt(b, a, data)

def local_filter(data, notch, width, flow, fhigh, fs):
    if (len(notch) > 1):
        for kk in range(len(notch)):
            data = notch_filter(data, notch[kk], width[kk], fs)
    else:
        data = notch_filter(data, notch, width, fs)
    data = elliptic_filter(data, flow, fhigh, fs)
    return signal.detrend(data)

def emg_activity(emg_amplitude, threshold, sampling_rate=1000):
    duration_min = int(0.05 * sampling_rate)
    threshold = threshold * np.std(emg_amplitude)
    activity = nk.signal_binarize(emg_amplitude, method="threshold", threshold=threshold)

    # Sanitize activity.
    info = _emg_activation_activations(activity, duration_min=duration_min)

    # Prepare Output.
    df_activity = nk.signal_formatpeaks(
        {"EMG_Activity": info["EMG_Activity"]},
        desired_length=len(emg_amplitude),
        peak_indices=info["EMG_Activity"],
    )
    df_onsets = nk.signal_formatpeaks(
        {"EMG_Onsets": info["EMG_Onsets"]},
        desired_length=len(emg_amplitude),
        peak_indices=info["EMG_Onsets"],
    )
    df_offsets = nk.signal_formatpeaks(
        {"EMG_Offsets": info["EMG_Offsets"]},
        desired_length=len(emg_amplitude),
        peak_indices=info["EMG_Offsets"],
    )

    # Modify output produced by signal_formatpeaks.
    for x in range(len(emg_amplitude)):
        if df_activity["EMG_Activity"][x] != 0:
            if df_activity.index[x] == df_activity.index.get_loc(x):
                df_activity["EMG_Activity"][x] = 1
            else:
                df_activity["EMG_Activity"][x] = 0
        if df_offsets["EMG_Offsets"][x] != 0:
            if df_offsets.index[x] == df_offsets.index.get_loc(x):
                df_offsets["EMG_Offsets"][x] = 1
            else:
                df_offsets["EMG_Offsets"][x] = 0

    activity_signal = pd.concat([df_activity, df_onsets, df_offsets], axis=1)

    return activity_signal, info

# =============================================================================
# Neurokit Internals
# =============================================================================
def _emg_activation_activations(activity, duration_min=0.05):

    activations = nk.events_find(
        activity, threshold=0.5, threshold_keep="above", duration_min=duration_min
    )
    activations["offset"] = activations["onset"] + activations["duration"]

    baseline = nk.events_find(
        activity == 0, threshold=0.5, threshold_keep="above", duration_min=duration_min
    )
    baseline["offset"] = baseline["onset"] + baseline["duration"]

    # Cross-comparison
    valid = np.isin(activations["onset"], baseline["offset"])
    onsets = activations["onset"][valid]
    offsets = activations["offset"][valid]

    # make sure offset indices are within length of signal
    offsets = offsets[offsets < len(activity)]

    new_activity = np.array([])
    for x, y in zip(onsets, offsets):
        activated = np.arange(x, y)
        new_activity = np.append(new_activity, activated)

    # Prepare Output.
    info = {"EMG_Onsets": onsets, "EMG_Offsets": offsets, "EMG_Activity": new_activity}

    return info