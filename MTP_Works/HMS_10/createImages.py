import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import multiprocessing as mp
from joblib import Parallel, delayed
from skimage.transform import resize
import timm

import os

BASE_DIR = "/home/m1/23CS60R76/MTP_Works/HMS_Brain_Activity/Harmful_Brain_Activity/"

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

df = pd.read_csv(f"{BASE_DIR}train.csv")

df["total_votes"] = df[["seizure_vote","lpd_vote","gpd_vote","lrda_vote","grda_vote","other_vote"]].sum(axis=1)

EEGid_label_list = df[["eeg_id","eeg_label_offset_seconds","seizure_vote","lpd_vote","gpd_vote","lrda_vote","grda_vote","other_vote","total_votes"]].values.tolist()
print("Total Samples:", len(EEGid_label_list))

print()
for x in EEGid_label_list:
    eeg_id, offset, sv, lpv, gpv, lrv, grv, ov, tv = x
    eeg_id = int(eeg_id)
    print(f"Processing EEG-ID:{eeg_id} with offset:{offset}")
    data_path = f'{BASE_DIR}train_eegs/{eeg_id}.parquet'
    img_path = f'{BASE_DIR}EEG_imgs/{eeg_id}_{offset}.npz'
    raw_50s_img = raweeg_50s_from_eeg_ver1(data_path, offset)
    raw_50s_img = resize(raw_50s_img, [518, 518])
    raw_50s_img = np.expand_dims(raw_50s_img, -1)
    votes_arr = np.array([sv, lpv, gpv, lrv, grv, ov])
    y = votes_arr/tv
    print("Image Shape:", raw_50s_img.shape)
    model = timm.create_model("vit_large_patch14_reg4_dinov2.lvd142m", num_classes=6, pretrained=True, in_chans=1).cuda()
    model.set_grad_checkpointing()
    raw_50s_imgs = np.array([raw_50s_img, raw_50s_img])
    raw_50s_imgs = raw_50s_imgs.transpose(1, 2).transpose(1, 3).contiguous()
    print('Number of parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    yPred = model(raw_50s_imgs)
    break