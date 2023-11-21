from enum import Enum
import pandas as pd
import neurokit2 as nk
from emg import emg_process

class Signal(Enum):
    ECG = 1
    PPG = 2
    EDA = 3
    RSP = 4
    EMG = 5

# use specific predefined functions (for signal preprocessing) in the neurokit2 lib
# return: preprocessed signal
def processor(signal, type, sampling_rate=1000):
    if type == Signal.ECG:
        processed_signal, _ = nk.ecg_process(signal, sampling_rate=sampling_rate)
    elif type == Signal.PPG:
        processed_signal, _ = nk.ppg_process(signal, sampling_rate=sampling_rate)
    elif type == Signal.EDA:
        processed_signal, _ = nk.eda_process(signal, sampling_rate=sampling_rate)
    elif type == Signal.RSP:
        processed_signal, _ = nk.rsp_process(signal, sampling_rate=sampling_rate)
        if processed_signal.RSP_Rate.isna().sum() > 0:
            processed_signal.RSP_Rate = nk.rsp_rate(processed_signal.RSP_Clean, method='xcorr')
    elif type == Signal.EMG:
        processed_signal, _ = emg_process(signal, sampling_rate=sampling_rate)

    return processed_signal

# create preprocessed signal windows with past samples of (window_size_past), current samples and future samples of (window_size_future)
# use specific predefined functions (for feature extraction) in the neurokit2 lib
# return: extracted features
def analyzer(signal, keypoints, type, window_size_past, window_size_future, sampling_rate=1000):
    df = None
    maximum = keypoints.max()
    for idx in keypoints:
        if idx - window_size_past < 0:
            onset = 0
            end = window_size_future / sampling_rate + idx / sampling_rate
        elif idx + window_size_future > maximum:
            onset = idx - window_size_past
            end = (maximum - idx) / sampling_rate + window_size_past / sampling_rate
        else:
            onset = idx - window_size_past
            end = (window_size_past + window_size_future + 1) / sampling_rate
        epoch = nk.epochs_create(signal, events=[onset], epochs_end=[end])
        
        if type == Signal.ECG:
            features = nk.ecg_analyze(epoch, sampling_rate=sampling_rate, method="event-related")
        elif type == Signal.PPG:
            features = nk.ppg_analyze(epoch, sampling_rate=sampling_rate, method="event-related")
        elif type == Signal.EDA:
            features = nk.eda_analyze(epoch, sampling_rate=sampling_rate, method="event-related")
        elif type == Signal.RSP:
            features = nk.rsp_analyze(epoch, sampling_rate=sampling_rate, method="event-related")
        elif type == Signal.EMG:
            try:
                features = nk.emg_analyze(epoch, sampling_rate=sampling_rate, method='event-related')
            except:
                epoch['1'].EMG_Onsets = 0
                features = nk.emg_analyze(epoch, sampling_rate=sampling_rate, method='event-related')
        
        df = features if df is None else pd.concat([df, features], axis=0)
    
    return df

# perform preprocessing and feature extractions
# return: preprocessed signals, extracted features
def feature_extractor(X, y, is_extract_features=True):
    signal = X['bvp']
    type = Signal.PPG
    bvp_signal = processor(signal, type)
    
    signal = X['ecg']
    type = Signal.ECG
    ecg_signal = processor(signal, type)
    
    signal = X['rsp']
    type = Signal.RSP
    rsp_signal = processor(signal, type)
    
    signal = X['gsr']
    type = Signal.EDA
    gsr_signal = processor(signal, type)

    type = Signal.EMG
    signal = X['emg_zygo']
    emg_zygo_signal = processor(signal, type)
    signal = X['emg_coru']
    emg_coru_signal = processor(signal, type)
    signal = X['emg_trap']
    emg_trap_signal = processor(signal, type)


    if is_extract_features:
        gsr = analyzer(gsr_signal, y.index, Signal.EDA, 2500, 2500)
        rsp = analyzer(rsp_signal, y.index, Signal.RSP, 4000, 4000)
        ecg = analyzer(ecg_signal, y.index, Signal.ECG, 10000, 10000)
        bvp = analyzer(bvp_signal, y.index, Signal.PPG, 5000, 5000)
        emg_zygo = analyzer(emg_zygo_signal, y.index, Signal.EMG, 200, 200)
        emg_coru = analyzer(emg_coru_signal, y.index, Signal.EMG, 200, 200)
        emg_trap = analyzer(emg_trap_signal, y.index, Signal.EMG, 200, 200)

        dropped_cols = ['Label', 'Event_Onset']
        emg_zygo = emg_zygo.drop(dropped_cols, axis=1)
        emg_coru = emg_coru.drop(dropped_cols, axis=1)
        emg_trap = emg_trap.drop(dropped_cols, axis=1)

        emg_zygo.columns = list(map(lambda s: s + '_zygo', list(emg_zygo.columns)))
        emg_coru.columns = list(map(lambda s: s + '_coru', list(emg_coru.columns)))
        emg_trap.columns = list(map(lambda s: s + '_trap', list(emg_trap.columns)))

        extracted_features = pd.concat([bvp.drop(dropped_cols, axis=1),
                              ecg.drop(dropped_cols, axis=1),
                              rsp.drop(dropped_cols, axis=1),
                              gsr.drop(dropped_cols, axis=1),
                              emg_zygo, emg_coru, emg_trap], axis=1).set_index(y.index)

    else:
        extracted_features = None

    emg_zygo_signal.columns = list(map(lambda s: s + '_zygo', list(emg_zygo_signal.columns)))
    emg_coru_signal.columns = list(map(lambda s: s + '_coru', list(emg_coru_signal.columns)))
    emg_trap_signal.columns = list(map(lambda s: s + '_trap', list(emg_trap_signal.columns)))

    processed_signal = pd.concat([ecg_signal, bvp_signal, gsr_signal, rsp_signal, emg_zygo_signal, emg_coru_signal, emg_trap_signal], axis=1).set_index(X.index)

    return processed_signal, extracted_features
