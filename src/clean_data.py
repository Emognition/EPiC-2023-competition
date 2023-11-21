import emg
import pandas as pd
import neurokit2 as nk
import numpy as np
from scipy import signal


#w, h = freqz(b, a, fs=1e3, worN=8000)
def butter_lowpass(cutoff, fs, order=5):
    return signal.butter(
        order, cutoff, fs=fs, btype='low', analog=False
    )


def filterTemperature(data, sampling_rate=1000):
    cutoff = 1.0
    offset = data[0]
    b, a = butter_lowpass(cutoff, sampling_rate)
    # remove offset from 0 before filtering to avoid large filter swinging
    y = signal.lfilter(b, a, (data-offset))
    return y+offset # re-apply offset


def clean(file: str):
    df = pd.read_csv(file)
    fs = 1000

    # process heart rate, blood volume pulse, skin conductance and breathing
    bio_df, bio_info = nk.bio_process(
        ecg=df.ecg,
        ppg=df.bvp,
        eda=df.gsr,
        rsp=df.rsp,
        sampling_rate=fs,
        keep=df.time
    )

    # process temperature (lowpass for cleaning and change (delta))
    tmp = filterTemperature(df.skt, fs)
    skt_df = pd.concat([
        df.skt,
        pd.DataFrame(
            # add 0 at beginning of delta to keep same length
            np.stack([tmp, np.concatenate([[0], np.diff(tmp)])], axis=1),
            columns=['SKT_Clean', 'SKT_Delta']
        )

    ], axis=1)

    # process emg
    emg_df1, emg_info1 = emg.emg_process(
        df.emg_zygo,
        sampling_rate=fs,
    )
    emg_df1.columns = [x + '_zygo' for x in emg_df1.columns]

    emg_df2, emg_info2 = emg.emg_process(
        df.emg_coru,
        sampling_rate=fs,
    )
    emg_df2.columns = [x + '_coru' for x in emg_df2.columns]

    emg_df3, emg_info3 = emg.emg_process(
        df.emg_trap,
        sampling_rate=fs,
    )
    emg_df3.columns = [x + '_trap' for x in emg_df3.columns]

    return pd.concat([bio_df, skt_df, emg_df1, emg_df2, emg_df3], axis=1)