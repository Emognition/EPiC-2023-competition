#%%
import os
import pandas as pd
import numpy as np
from scipy import interpolate
from scipy import interpolate
import neurokit2 as nk
from biosppy.signals import ecg
from hrvanalysis import remove_outliers
from hrvanalysis import remove_ectopic_beats
from hrvanalysis import interpolate_nan_values
from hrvanalysis import get_time_domain_features
from biosppy.signals import ecg
import cvxopt as cv
from biosppy.signals import ecg
from joblib import Parallel, delayed
from functools import partial

def cvxEDA_pyEDA(y, delta, tau0=2., tau1=0.7, delta_knot=10., alpha=8e-4, gamma=1e-2,
           solver=None, options={'reltol':1e-9}):
    """CVXEDA Convex optimization approach to electrodermal activity processing
    This function implements the cvxEDA algorithm described in "cvxEDA: a
    Convex Optimization Approach to Electrodermal Activity Processing"
    (http://dx.doi.org/10.1109/TBME.2015.2474131, also available from the
    authors' homepages).
    Arguments:
       y: observed EDA signal (we recommend normalizing it: y = zscore(y))
       delta: sampling interval (in seconds) of y
       tau0: slow time constant of the Bateman function
       tau1: fast time constant of the Bateman function
       delta_knot: time between knots of the tonic spline function
       alpha: penalization for the sparse SMNA driver
       gamma: penalization for the tonic spline coefficients
       solver: sparse QP solver to be used, see cvxopt.solvers.qp
       options: solver options, see:
                http://cvxopt.org/userguide/coneprog.html#algorithm-parameters
    Returns (see paper for details):
       r: phasic component
       p: sparse SMNA driver of phasic component
       t: tonic component
       l: coefficients of tonic spline
       d: offset and slope of the linear drift term
       e: model residuals
       obj: value of objective function being minimized (eq 15 of paper)
    """
    
    n = len(y)
    y = cv.matrix(y)

    # bateman ARMA model
    a1 = 1./min(tau1, tau0) # a1 > a0
    a0 = 1./max(tau1, tau0)
    ar = np.array([(a1*delta + 2.) * (a0*delta + 2.), 2.*a1*a0*delta**2 - 8.,
        (a1*delta - 2.) * (a0*delta - 2.)]) / ((a1 - a0) * delta**2)
    ma = np.array([1., 2., 1.])

    # matrices for ARMA model
    i = np.arange(2, n)
    A = cv.spmatrix(np.tile(ar, (n-2,1)), np.c_[i,i,i], np.c_[i,i-1,i-2], (n,n))
    M = cv.spmatrix(np.tile(ma, (n-2,1)), np.c_[i,i,i], np.c_[i,i-1,i-2], (n,n))

    # spline
    delta_knot_s = int(round(delta_knot / delta))
    spl = np.r_[np.arange(1.,delta_knot_s), np.arange(delta_knot_s, 0., -1.)] # order 1
    spl = np.convolve(spl, spl, 'full')
    spl /= max(spl)
    # matrix of spline regressors
    i = np.c_[np.arange(-(len(spl)//2), (len(spl)+1)//2)] + np.r_[np.arange(0, n, delta_knot_s)]
    nB = i.shape[1]
    j = np.tile(np.arange(nB), (len(spl),1))
    p = np.tile(spl, (nB,1)).T
    valid = (i >= 0) & (i < n)
    B = cv.spmatrix(p[valid], i[valid], j[valid])

    # trend
    C = cv.matrix(np.c_[np.ones(n), np.arange(1., n+1.)/n])
    nC = C.size[1]

    # Solve the problem:
    # .5*(M*q + B*l + C*d - y)^2 + alpha*sum(A,1)*p + .5*gamma*l'*l
    # s.t. A*q >= 0

    old_options = cv.solvers.options.copy()
    cv.solvers.options.clear()
    cv.solvers.options.update(options)
    if solver == 'conelp':
        # Use conelp
        z = lambda m,n: cv.spmatrix([],[],[],(m,n))
        G = cv.sparse([[-A,z(2,n),M,z(nB+2,n)],[z(n+2,nC),C,z(nB+2,nC)],
                    [z(n,1),-1,1,z(n+nB+2,1)],[z(2*n+2,1),-1,1,z(nB,1)],
                    [z(n+2,nB),B,z(2,nB),cv.spmatrix(1.0, range(nB), range(nB))]])
        h = cv.matrix([z(n,1),.5,.5,y,.5,.5,z(nB,1)])
        c = cv.matrix([(cv.matrix(alpha, (1,n)) * A).T,z(nC,1),1,gamma,z(nB,1)])
        res = cv.solvers.conelp(c, G, h, dims={'l':n,'q':[n+2,nB+2],'s':[]})
        obj = res['primal objective']
    else:
        # Use qp
        Mt, Ct, Bt = M.T, C.T, B.T
        H = cv.sparse([[Mt*M, Ct*M, Bt*M], [Mt*C, Ct*C, Bt*C], 
                    [Mt*B, Ct*B, Bt*B+gamma*cv.spmatrix(1.0, range(nB), range(nB))]])
        f = cv.matrix([(cv.matrix(alpha, (1,n)) * A).T - Mt*y,  -(Ct*y), -(Bt*y)])
        res = cv.solvers.qp(H, f, cv.spmatrix(-A.V, A.I, A.J, (n,len(f))),
                            cv.matrix(0., (n,1)), solver=solver)
        obj = res['primal objective'] + .5 * (y.T * y)
    cv.solvers.options.clear()
    cv.solvers.options.update(old_options)

    l = res['x'][-nB:]
    d = res['x'][n:n+nC]
    t = B*l + C*d
    q = res['x'][:n]
    p = A * q
    r = M * q
    e = y - r - t

    return (np.array(a).ravel() for a in (r, p, t, l, d, e, obj))


def get_nn(peaks, fs):
    """Convert beat peaks in samples to NN intervals and timestamp."""
    rr = np.diff(peaks, prepend=0) * 1000 / fs

    # This remove outliers from signal
    rr = remove_outliers(rr, low_rri=300, high_rri=2000, verbose=False)
    # This replace outliers nan values with linear interpolation
    rr = interpolate_nan_values(rr, interpolation_method="linear")

    # This remove ectopic beats from signal
    # TODO: esto puede no tener sentido en PPG, pero los metodos de features
    #  estan basados en NN y no en RR.
    rr = remove_ectopic_beats(rr, method="malik", verbose=False)
    # This replace ectopic beats nan values with linear interpolation
    rr = np.array(interpolate_nan_values(rr))

    rr[np.where(np.isnan(rr))] = 0
    
    return np.array(rr)

def process_ecg(signal, fs):
    """Get NN interval from ecg signal."""
    _, _, rpeaks, _, _, _, _ = ecg.ecg(signal,
                                       sampling_rate=fs,
                                       show=False)
    rr = get_nn(rpeaks, fs)

    return rr, rpeaks


def moving_hrv(rr, window_size, step_size, fs):
    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)
    hrv = []
    for i in range(0, len(rr) - window_samples, step_samples):
        rr_window = rr[i:i + window_samples]
        features = get_time_domain_features(rr_window)
        hrv.append(features['rmssd'])
    return hrv

def preprocess_physiology(data):
    
    index = data.index
    fs = 1000
    
    ecg_cleaned = nk.ecg_clean(data["ecg"])
    ecg_signals = pd.DataFrame({"ecg_cleaned": ecg_cleaned}, index=index)
    
    rr, peaks = process_ecg(data.ecg, fs)

    # Interpolate RR
    interpf = interpolate.interp1d(peaks/fs, rr, bounds_error=False,
                                fill_value=(rr[0], rr[-1]))

    # Create a time vector with the same length as the index
    timestamp = np.linspace(0, len(data["ecg"]) / fs, len(index))

    # Apply the interpolating function to the time vector
    rr_signal = interpf(timestamp)

    rr_signals = pd.DataFrame({"rr_signal": rr_signal}, index=index)
        
    # Preprocess BVP signal
    bvp_cleaned = nk.ppg_clean(data["bvp"])
    bvp_signals = pd.DataFrame({"bvp_cleaned": bvp_cleaned}, index=index)

    # Preprocess GSR signal
    gsr_cleaned = nk.eda_clean(data["gsr"], method='BioSPPy')
    gsr_signals = pd.DataFrame({"gsr_cleaned": gsr_cleaned}, index=index)

    # Calculate phasic and tonic components of GSR signal
    gsr_components = nk.eda_phasic(gsr_cleaned, sampling_rate=1000, method='cvxEDA')
    gsr_components.index = index
    gsr_components.columns = ['gsr_tonic', 'gsr_phasic']
    
    # Extrae Phasic, Tonic y SMNA activity de la señal GSR
    gsr_cleaned_crop = np.array_split(gsr_cleaned, 20)
    cvx = np.empty([1,0])
    for i in range(len(gsr_cleaned_crop)):
        [_, p, _, _ , _ , _ , _] = cvxEDA_pyEDA(gsr_cleaned_crop[i], 1./1000)
        p = p[np.newaxis, :]  # Add a new axis to make p a 2D array with 1 row
        #if i == len(gsr_cleaned_crop)-1:
            #phasic_gsr = np.zeros(len(gsr_cleaned_crop[i]))  # llenar con ceros
            #tonic_gsr = np.full(len(gsr_cleaned_crop[i]), cvx_aux[2][-1])  # llenar con el último valor del fragmento anterior
        #cvx_aux = np.vstack((phasic_gsr, p, tonic_gsr))
        cvx = np.hstack((cvx, p))

    #eda_phasic = cvx[0]
    eda_smna = cvx[0]
    #eda_tonic = cvx[2]

    # Guarda la actividad del nervio sudomotor en gsr_SMNA
    gsr_SMNA = eda_smna
    #gsr_phasic = eda_phasic
    #gsr_tonic = eda_tonic

    # Agrega gsr_SMNA a la variable data
    gsr_SMNA = pd.DataFrame(gsr_SMNA, columns=["gsr_SMNA"], index=index)
    #gsr_phasic = pd.DataFrame(gsr_phasic, columns=["gsr_phasic"], index=index)
    #gsr_tonic = pd.DataFrame(gsr_tonic, columns=["gsr_tonic"], index=index)

    # Preprocess RSP signal
    rsp_cleaned = nk.rsp_clean(data["rsp"])
    rsp_signals = pd.DataFrame({"rsp_cleaned": rsp_cleaned}, index=index)
    fs =1000
    
    df_peaks, peaks_dict = nk.rsp_peaks(rsp_cleaned, sampling_rate=fs)
    info = nk.rsp_fixpeaks(peaks_dict)
    formatted = nk.signal_formatpeaks(info, desired_length=len(rsp_cleaned),peak_indices=info["RSP_Peaks"])
    formatted['RSP_Clean'] = rsp_cleaned
    
    # Extract rate
    rsp_rate = nk.rsp_rate(formatted, sampling_rate=fs)
    rsp_rate_df = pd.DataFrame(rsp_rate, columns=["resp_rate"], index=index)

    # Preprocess EMG ZYGO signal
    emg_zygo_cleaned = nk.emg_clean(data["emg_zygo"])        
    emg_zygo_signals = pd.DataFrame({"emg_zygo_cleaned": emg_zygo_cleaned}, index=index)

    # Preprocess EMG CORU signal
    emg_coru_cleaned = nk.emg_clean(data["emg_coru"])
    emg_coru_signals = pd.DataFrame({"emg_coru_cleaned": emg_coru_cleaned}, index=index)

    # Preprocess EMG TRAP signal
    emg_trap_cleaned = nk.emg_clean(data["emg_trap"])
    emg_trap_signals = pd.DataFrame({"emg_trap_cleaned": emg_trap_cleaned}, index=index)

    skt_signals = pd.DataFrame({"skt_filtered": data["skt"]}, index=index)

    # Combine preprocessed signals into one DataFrame
    preprocessed_data = pd.concat([ecg_signals,
                                    rr_signals,
                                    bvp_signals,
                                    gsr_signals,
                                    gsr_components,
                                    #gsr_tonic,
                                    #gsr_phasic,
                                    gsr_SMNA,
                                    rsp_signals,
                                    rsp_rate_df,
                                    emg_zygo_signals,
                                    emg_coru_signals,
                                    emg_trap_signals,
                                    skt_signals], axis=1)

    return preprocessed_data

def process_file(file, root, src_path, dst_path):
    file_path = os.path.join(root, file)
    data = pd.read_csv(file_path, index_col="time")
    preprocessed_data = preprocess_physiology(data)
    preprocessed_file_path = file_path.replace(src_path, dst_path)
    os.makedirs(os.path.dirname(preprocessed_file_path), exist_ok=True)
    preprocessed_data.to_csv(preprocessed_file_path, index=True)

def file_already_processed(filename, root, dst_path):
    # Get the relative path of the source file
    rel_path = os.path.relpath(root, source_dir)

    # Construct the corresponding preprocessed file path
    preprocessed_file_path = os.path.join(dst_path, rel_path, filename)
    return os.path.exists(preprocessed_file_path)

def process_files_in_physiology(src_path, dst_path, test_mode=False):
    outer_loop_break = False  # Add this flag variable to break the outer loop
    for root, _, files in os.walk(src_path):
        if os.path.basename(root) == "physiology":
            parent_directory = os.path.dirname(root)
            csv_files = [file for file in files if file.endswith(".csv")]

            if test_mode:
                csv_files = csv_files[:1]

            # Use partial to create a new function with root and dst_path as fixed arguments
            check_file_already_processed = partial(file_already_processed, root=root, dst_path=dst_path)

            # Filter out the files that have already been processed
            csv_files = [file for file in csv_files if not check_file_already_processed(file)]

            Parallel(n_jobs=-1)(delayed(process_file)(file, root, src_path, dst_path) for file in csv_files)

            if test_mode:
                outer_loop_break = True  # Set the flag to True when you want to break the outer loop
                break
        if outer_loop_break:  # Check the flag in the outer loop, and break if it's True
            break

source_dir = "../data/raw"
destination_dir = "../data/preprocessed"
test_mode = False  # Set this to False if you want to run the script for all participants

process_files_in_physiology(source_dir, destination_dir, test_mode)

# %%
