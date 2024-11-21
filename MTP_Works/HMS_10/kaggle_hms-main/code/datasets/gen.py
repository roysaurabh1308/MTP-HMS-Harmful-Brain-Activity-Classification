import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed

import os

RANDOM_SEED = 1086

NAMES = ['LL','LP','RP','RR']

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]


from scipy import signal

SFREQ = 200

filter_range = [0.5, 40]
b, a = signal.butter(3, np.float32(filter_range)*2/SFREQ, 'bandpass')

def stft_spec_from_50s_eeg_ver1(parquet_path, eeg_label_offset_seconds):
    EEG_LENGTH = 50
    eeg = pd.read_parquet(parquet_path)
    
    time_temp = eeg_label_offset_seconds*200
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200) 
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    
    eeg = eeg.iloc[time_start: time_stop]
    
    list_eeg = list()
    for k in range(4):
        COLS = FEATS[k]
        img = np.zeros((128,142,4),dtype='float32')
        for kk in range(4):
            eeg_1 = eeg[COLS[kk]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values
            
            eeg_2 = eeg[COLS[kk+1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values
            
            new_eeg = eeg_1 - eeg_2
            del eeg_1
            del eeg_2
            # new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            fs = 200  
            nperseg = 70
            noverlap = 0
            f, t, spec = signal.spectrogram(new_eeg, fs, nperseg=nperseg, noverlap=noverlap, nfft=256)
            # print(spec.shape)
            
            spec = np.abs(spec) 
            spec = np.log1p(spec).astype("float32")
            # plt.pcolormesh(t, f, spec)
            # plt.set_cmap('jet')
            # plt.axis('off')

            # plt.savefig("eeg_spec_butter", bbox_inches='tight', pad_inches=0, dpi=35)
            # plt.clf()

            img[:,:,kk] += spec[:128, :]
        img = np.concatenate((img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3]), 1)
        # print(img.shape)
        list_eeg.append(img)
    img = np.concatenate(list_eeg, 0)    
    # img = np.concatenate((img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3]), 0)
    img /= 2
    return img

def stft_spec_from_50s_eeg_ver2(parquet_path, eeg_label_offset_seconds):
    EEG_LENGTH = 50
    eeg = pd.read_parquet(parquet_path)
    
    time_temp = eeg_label_offset_seconds*200
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200) 
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    
    eeg = eeg.iloc[time_start: time_stop]
    
    list_eeg = list()
    for k in range(4):
        COLS = FEATS[k]
        img = np.zeros((128,256,4),dtype='float32')
        for kk in range(4):
            eeg_1 = eeg[COLS[kk]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values
            
            eeg_2 = eeg[COLS[kk+1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values
            
            new_eeg = eeg_1 - eeg_2
            del eeg_1
            del eeg_2
            # new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            fs = 200  
            nperseg = 39
            noverlap = 0
            f, t, spec = signal.spectrogram(new_eeg, fs, nperseg=nperseg, noverlap=noverlap, nfft=256)
            # print(spec.shape)
            
            spec = np.abs(spec) 
            spec = np.log1p(spec).astype("float32")
            # plt.pcolormesh(t, f, spec)
            # plt.set_cmap('jet')
            # plt.axis('off')

            # plt.savefig("eeg_spec_butter", bbox_inches='tight', pad_inches=0, dpi=35)
            # plt.clf()

            img[:,:,kk] += spec[:128, :]
        img = np.concatenate((img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3]), 1)
        # print(img.shape)
        list_eeg.append(img)
    img = np.concatenate(list_eeg, 0)    
    # img = np.concatenate((img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3]), 0)
    img /= 2
    return img

def stft_spec_from_50s_eeg_ver3(parquet_path, eeg_label_offset_seconds):
    EEG_LENGTH = 50
    eeg = pd.read_parquet(parquet_path)
    
    time_temp = eeg_label_offset_seconds*200
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200) 
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    
    eeg = eeg.iloc[time_start: time_stop]
    
    img = np.zeros((128,256,4),dtype='float32')
    for k in range(4):
        COLS = FEATS[k]
        for kk in range(4):
            eeg_1 = eeg[COLS[kk]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values
            
            eeg_2 = eeg[COLS[kk+1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values
            
            new_eeg = eeg_1 - eeg_2
            del eeg_1
            del eeg_2
            # new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            fs = 200  
            nperseg = len(new_eeg)//256
            noverlap = 0
            f, t, spec = signal.stft(new_eeg, fs, nperseg=nperseg, noverlap=noverlap, nfft=256)

            spec = np.abs(spec) 
            spec = np.log1p(spec).astype("float32")

            img[:,:,k] += spec[:128, 1:257]
        img[:,:,k] /= 4
    img = np.concatenate((img[:,:,0], img[:,:,1], img[:,:,2], img[:,:,3]), 0)
    return img

RAW_FEATURES = {'LL': ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1'],
         'RL': ['Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2'],
         'LP': ['Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1'],
         'RP': ['Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2']}

def raweeg_50s_from_eeg_ver1(parquet_path, eeg_label_offset_seconds):
    EEG_LENGTH = 50
    raw_eeg = pd.read_parquet(parquet_path)
    time_temp = eeg_label_offset_seconds*200
    time_start =  round(time_temp + (50 - EEG_LENGTH) / 2 * 200) 
    time_stop =  round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg_default = raw_eeg.loc[time_start: (time_stop - 1), :].reset_index(drop=True)
    list_eeg = list()
    for region in RAW_FEATURES.keys():

        eeg = np.zeros((len(RAW_FEATURES[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATURES[region]):
            eeg_1 = eeg_default.loc[:, chan.split('-')[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values
            
            eeg_2 = eeg_default.loc[:, chan.split('-')[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values
            
            new_eeg = eeg_1 - eeg_2
            del eeg_1
            del eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32")
            eeg[chan_i, :] = new_eeg
        
        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate((eeg[0,:,:], eeg[1,:,:], eeg[2,:,:], eeg[3,:,:]), 1)
        list_eeg.append(eeg)

    eeg = np.concatenate(list_eeg, 1)
    eeg /= 104
    
    return eeg

def raweeg_50s_from_eeg_ver2(parquet_path, eeg_label_offset_seconds):
    EEG_LENGTH = 20
    raw_eeg = pd.read_parquet(parquet_path)
    time_temp = eeg_label_offset_seconds*200
    time_start =  round(time_temp + (50 - EEG_LENGTH) / 2 * 200) 
    time_stop =  round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg_default = raw_eeg.loc[time_start: (time_stop - 1), :].reset_index(drop=True)
    list_eeg = list()
    for region in RAW_FEATURES.keys():

        eeg = np.zeros((len(RAW_FEATURES[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATURES[region]):
            eeg_1 = eeg_default.loc[:, chan.split('-')[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values
            
            eeg_2 = eeg_default.loc[:, chan.split('-')[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values
            
            new_eeg = eeg_1 - eeg_2
            del eeg_1
            del eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32")
            eeg[chan_i, :] = new_eeg
        
        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate((eeg[0,:,:], eeg[1,:,:], eeg[2,:,:], eeg[3,:,:]), 1)
        list_eeg.append(eeg)

    eeg = np.concatenate(list_eeg, 1)
    eeg /= 104
    
    return eeg

def raweeg_10s_from_eeg(parquet_path, eeg_label_offset_seconds):
    EEG_LENGTH = 10
    raw_eeg = pd.read_parquet(parquet_path)
    time_temp = eeg_label_offset_seconds*200
    time_start =  round(time_temp + (50 - EEG_LENGTH) / 2 * 200) 
    time_stop =  round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg_default = raw_eeg.loc[time_start: (time_stop - 1), :].reset_index(drop=True)
    list_eeg = list()
    for region in RAW_FEATURES.keys():

        eeg = np.zeros((len(RAW_FEATURES[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATURES[region]):
            eeg_1 = eeg_default.loc[:, chan.split('-')[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values
            
            eeg_2 = eeg_default.loc[:, chan.split('-')[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values
            
            new_eeg = eeg_1 - eeg_2
            del eeg_1
            del eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32")
            eeg[chan_i, :] = new_eeg
        
        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate((eeg[0,:,:], eeg[1,:,:], eeg[2,:,:], eeg[3,:,:]), 1)
        list_eeg.append(eeg)

    eeg = np.concatenate(list_eeg, 1)
    eeg /= 104
    
    return eeg
