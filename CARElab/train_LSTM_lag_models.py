import os

if __name__ == '__main__':
    # train 1 model per physiological signal and all shifts
    # for 25 epochs with 50 input samples. 128 hidden units and 2 outputs
    for sensor in ['ecg', 'bvp', 'gsr', 'rsp', 'skt', 'emg_zygo', 'emg_coru', 'emg_trap']:
        for offset in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
            cmd = f'python3 train_LSTM.py 50 128 2 {sensor} -o {offset} -e 25'
            print(cmd)
            os.system(cmd)
