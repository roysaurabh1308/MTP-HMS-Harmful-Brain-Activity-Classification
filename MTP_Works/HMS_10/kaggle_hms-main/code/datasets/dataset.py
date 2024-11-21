import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
import albumentations as A
import torchaudio.transforms as T
import torchvision

from datasets.gen import stft_spec_from_50s_eeg_ver1, stft_spec_from_50s_eeg_ver2, stft_spec_from_50s_eeg_ver3, raweeg_50s_from_eeg_ver1, raweeg_50s_from_eeg_ver2, raweeg_10s_from_eeg
import random
import math
import json
import pandas as pd
from skimage.transform import resize

BASE_DIR = "/home/m1/23CS60R76/MTP_Works/HMS_Brain_Activity/Harmful_Brain_Activity/"

def random_time_crop(image):
    if len(image.shape) == 3:
        freq_w, time_w, _ = image.shape
    else:
        freq_w, time_w = image.shape
    prob = random.random()
    pad_l = random.uniform(0.0, 0.3)
    pad_r = random.uniform(0.0, 0.3)
    crop_l = int(time_w*pad_l)
    crop_r = int(time_w*pad_r)
    x_min = crop_l
    x_max = time_w - crop_r
    if prob < 0.5:
        if len(image.shape) == 3:
            image = image[:, x_min:x_max,:]
        else:
            image = image[:, x_min:x_max]

    return image

class ImageFolder(data.Dataset):
    def __init__(self, df, default_configs, randaug_magnitude, mode):
        super().__init__()
        # self.eeg_data_path = "/home/datasets/full_eeg_spectrograms/"
        self.eeg_spec_type = default_configs["eeg_spec_type"]
        self.raw_50_type = default_configs["raw_50_type"]
        self.spec_data_path = f"{BASE_DIR}train_spectrograms/"
        self.raw_50s_data_path = "/home/datasets/full_raw_eeg_50s"
        self.raw_10s_data_path = "/home/datasets/full_raw_eeg_10s"
        # self.raw_1d_data_path = "/home/datasets/raw_eeg_1d"
        self.df = df.reset_index(drop=True)
        self.eed_ids = df.groupby(["eeg_id"]).head(1)["eeg_id"].values

        self.mode = mode
        self.randaug_magnitude = randaug_magnitude

        self.train_imgsize = default_configs["train_image_size"]
        self.test_imgsize = default_configs["test_image_size"]

        
        params4 = {    
            "num_masks_x": (2, 4),
            "num_masks_y": (2, 4),    
            "mask_y_length": 8,
            "mask_x_length": (10, 20),
            "fill_value": 0,  

        }
        self.train_spec_transform = A.Compose([
            # A.Resize(height=self.train_imgsize[0], width=self.train_imgsize[1], p=1),
            A.XYMasking(**params4, p=0.5)
        ]
        )
        

        # self.train_raw_transform = A.Compose([
        #     A.Resize(height=self.train_imgsize[0], width=self.train_imgsize[1], p=1),
        # ]
        # )
       
        # self.test_transform = A.Compose([
        #     A.Resize(height=self.test_imgsize[0], width=self.test_imgsize[1], p=1)
        # ]
        # )

    def __len__(self):
        if self.mode == 'gen_softlabel':
            return len(self.df)
        else:
            return len(self.eed_ids)

    def __getitem__(self, index):
        
        if self.mode == 'gen_softlabel':
            row = self.df.loc[index]
            eeg_id = row.eeg_id
        else:
            eeg_id = self.eed_ids[index]
            row = self.df.loc[self.df["eeg_id"]==eeg_id].sample(random_state=42).iloc[0]
        # print("rank: ", torch.distributed.get_rank(), row)
        eeg_id = str(row.eeg_id)
        spec_id = str(row.spec_id)
        eeg_label_offset_seconds = row.eeg_label_offset_seconds
        spec_image_path = os.path.join(self.spec_data_path, spec_id + ".parquet")
        # eeg_image_path = os.path.join(self.eeg_data_path, eeg_id + "_" + str(eeg_label_offset_seconds) + ".npy")
        raw_50s_image_path = os.path.join(self.raw_50s_data_path, eeg_id + "_" + str(eeg_label_offset_seconds) + ".npy")
        raw_10s_image_path = os.path.join(self.raw_10s_data_path, eeg_id + "_" + str(eeg_label_offset_seconds) + ".npy")
        
        spec = pd.read_parquet(spec_image_path)
        spec_img = spec.values[:, 1:].T.astype("float32")
        spec_offset = int(row["spectrogram_label_offset_seconds"])
        # if self.mode == "train":
        #     spec_offset = max(spec_offset + random.randint(-1, 1), 0)

        spec_offset = spec_offset // 2
        # print(spec_offset)
        spec_img = spec_img[:, spec_offset: spec_offset + 300]
        PATH = f"{BASE_DIR}train_eegs/"

        # raw_50s_img = np.load(raw_50s_image_path).astype("float32")
        # raw_10s_img = np.load(raw_10s_image_path).astype("float16")
        if self.raw_50_type == "ver1":
            raw_50s_img = raweeg_50s_from_eeg_ver1(f'{PATH}{eeg_id}.parquet', int(eeg_label_offset_seconds))
        elif self.raw_50_type == "ver2":
            raw_50s_img = raweeg_50s_from_eeg_ver2(f'{PATH}{eeg_id}.parquet', int(eeg_label_offset_seconds))

        raw_10s_img = raweeg_10s_from_eeg(f'{PATH}{eeg_id}.parquet', int(eeg_label_offset_seconds))
        if self.mode == "train":
            if np.random.rand() < 0.5:
                for i in range(4):
                    for j in range(3):
                        k_i = random.randint(0, 49)
                        if np.random.rand() < 0.5:
                            raw_50s_img[j*50:(j+1)*50, i*50+k_i] = 0
            if np.random.rand() < 0.5:
                for i in range(16):
                    j = random.randint(0, 9)
                    if np.random.rand() < 0.5:
                        raw_10s_img[:, i*10 +j] = 0

        # eeg_img = np.load(eeg_image_path).astype("float16")
        
        
        # if self.mode == "train":
        #     new_eeg_label_offset_seconds = max(int(eeg_label_offset_seconds) + random.randint(-1, 1), 0)
        # else:
        new_eeg_label_offset_seconds = int(eeg_label_offset_seconds)

        if self.eeg_spec_type == "ver1":
            eeg_img = stft_spec_from_50s_eeg_ver1(f'{PATH}{eeg_id}.parquet', new_eeg_label_offset_seconds)
            eeg_img = resize(eeg_img, self.train_imgsize)
        elif self.eeg_spec_type == "ver2":
            eeg_img = stft_spec_from_50s_eeg_ver2(f'{PATH}{eeg_id}.parquet', new_eeg_label_offset_seconds)
        elif self.eeg_spec_type == "ver3":
            eeg_img = stft_spec_from_50s_eeg_ver3(f'{PATH}{eeg_id}.parquet', new_eeg_label_offset_seconds)
            eeg_img = resize(eeg_img, self.train_imgsize)
        # raw_1d = np.load(raw_1d_path)

        eps = 1e-6

        spec_img = np.clip(spec_img,np.exp(-4),np.exp(8))
        spec_img = np.log(spec_img)
        spec_img = np.nan_to_num(spec_img, nan=0.0) 

        
        spec_img = resize(spec_img, self.train_imgsize)
        raw_10s_img = resize(raw_10s_img, self.train_imgsize)
        raw_50s_img = resize(raw_50s_img, self.train_imgsize)
        eeg_img = np.expand_dims(eeg_img, -1)
        spec_img = np.expand_dims(spec_img, -1)
        raw_50s_img = np.expand_dims(raw_50s_img, -1)
        raw_10s_img = np.expand_dims(raw_10s_img, -1)
        
        if self.mode == 'train': 
            spec_img = self.train_spec_transform(image=spec_img)["image"]
            eeg_img = self.train_spec_transform(image=eeg_img)["image"]

            # raw_50s_img = self.train_raw_transform(image=raw_50s_img)["image"]
            # raw_10s_img = self.train_raw_transform(image=raw_10s_img)["image"]
        # else:
        #     eeg_img = self.test_transform(image=eeg_img)["image"]
        #     spec_img = self.test_transform(image=spec_img)["image"]
        #     raw_50s_img = self.test_transform(image=raw_50s_img)["image"]
        #     raw_10s_img = self.test_transform(image=raw_10s_img)["image"]

        if self.eeg_spec_type == "ver3":
            img_mean = eeg_img.mean(axis=(0, 1))
            img_std = eeg_img.std(axis=(0, 1))
            eeg_img = (eeg_img - img_mean) / (img_std + eps)
        
        img_mean = spec_img.mean(axis=(0, 1))
        img_std = spec_img.std(axis=(0, 1))
        spec_img = (spec_img - img_mean) / (img_std + eps)

        # img_mean = raw_50s_img.mean(axis=(0, 1))
        # img_std = raw_50s_img.std(axis=(0, 1))
        # raw_50s_img = (raw_50s_img - img_mean) / (img_std + eps)

        # img_mean = raw_10s_img.mean(axis=(0, 1))
        # img_std = raw_10s_img.std(axis=(0, 1))
        # raw_10s_img = (raw_10s_img - img_mean) / (img_std + eps)

        label = row.label
        label = np.array([float(num) for num in label.split(' ')])
        
        # print(img.dtype)
        if self.mode == 'gen_softlabel':
            spec_id = row.spec_id
            eeg_offset = row.eeg_label_offset_seconds
            spec_offset = row["spectrogram_label_offset_seconds"]
            return spec_img, eeg_img, raw_50s_img, raw_10s_img, label, eeg_id, spec_id, eeg_offset, spec_offset
        elif self.mode == 'test':
            num_vote = row.total_votes
            return spec_img, eeg_img, raw_50s_img, raw_10s_img, label, eeg_id, num_vote
        else:
            return spec_img, eeg_img, raw_50s_img, raw_10s_img, label, eeg_id


