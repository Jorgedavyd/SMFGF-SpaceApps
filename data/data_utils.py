import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np
def min_to_hour(idx):
    return int(np.floor(idx/60))
def hour_to_3_hour(idx):
    return int(np.floor(idx/3))

def map_to_kp_index(interval):
    if interval < 0 or interval > 28:
        raise ValueError("Interval must be in the range 0 to 28")

    # Define the mapping from interval to KP index with minus and plus signs
    mapping = {
        0: "0", 1: "0+", 2: "1-", 3: "1", 4: "1+", 5: "2-", 6: "2", 7: "2+", 8: "3-", 9: "3",
        10: "3+", 11: "4-", 12: "4", 13: "4+", 14: "5-", 15: "5", 16: "5+", 17: "6-", 18: "6",
        19: "6+", 20: "7-", 21: "7", 22: "7+", 23: "8-", 24: "8", 25: "8+", 26: "9-", 27: "9", 28: "9+"
    }

    return mapping[interval]

def map_kp_index_to_interval(kp_index):
    # Define the mapping from KP index to interval
    mapping = {
        "0": 0, "0+": 1, "1-": 2, "1": 3, "1+": 4, "2-": 5, "2": 6, "2+": 7, "3-": 8, "3": 9,
        "3+": 10, "4-": 11, "4": 12, "4+": 13, "5-": 14, "5": 15, "5+": 16, "6-": 17, "6": 18,
        "6+": 19, "7-": 20, "7": 21, "7+": 22, "8-": 23, "8": 24, "8+": 25, "9-": 26, "9": 27, "9+": 28
    }

    if kp_index in mapping:
        return mapping[kp_index]
    else:
        raise ValueError(f"Invalid KP index: {kp_index}")
    


#We need the pred_length to be of size divisible by 3 if possible
class RefinedTrainingDataset(Dataset):
    def __init__(self, l1_df, l2_df, dst_series, kp_series, sequence_length, prediction_length, hour = False):
        #l1 scaler
        self.x_scaler = StandardScaler()
        self.raw = self.x_scaler.fit_transform(l1_df.values)
        #l2 scaler
        self.x_hat_scaler = StandardScaler()
        self.pro = self.x_hat_scaler.fit_transform(l2_df.values)
        #dst scaler
        self.dst_scaler = StandardScaler() #
        self.dst = self.dst_scaler.fit_transform(dst_series.values.reshape(-1,1))
        #Kp scaler
        self.kp = kp_series.apply(map_kp_index_to_interval).values.reshape(-1,1)
        #other parameters
        self.sequence_length = sequence_length
        self.mode = hour
        self.pred_length = prediction_length #this will define how many hours later we want to train our model on 
    def __len__(self):
        return self.raw.shape[0] - (self.sequence_length + self.pred_length) + 1
    def __getitem__(self, idx):
        l1_sample = self.raw[idx:idx+self.sequence_length, :]
        l2_sample = self.pro[idx:idx+self.sequence_length, :]
        if self.mode:
            dst = self.dst[idx+self.sequence_length:idx+self.sequence_length+self.pred_length]
            kp = self.kp[hour_to_3_hour(idx+self.sequence_length):hour_to_3_hour(idx+self.sequence_length) + hour_to_3_hour(self.pred_length)]
        else:
            dst = self.dst[min_to_hour(idx+self.sequence_length):min_to_hour(idx+self.sequence_length)+self.pred_length]
            kp = self.kp[hour_to_3_hour(min_to_hour(idx+self.sequence_length)):hour_to_3_hour(min_to_hour(idx+self.sequence_length))+hour_to_3_hour(min_to_hour(self.pred_length))]
        l1_sample = torch.tensor(l1_sample, dtype=torch.float32)
        l2_sample = torch.tensor(l2_sample, dtype=torch.float32)
        dst = torch.tensor(dst, dtype=torch.float32)
        kp = torch.tensor(kp, dtype=torch.float32)
        return l1_sample, l2_sample, dst, kp

class NormalTrainingDataset(Dataset):
    def __init__(self, l1_df, dst_series, kp_series, sequence_length, prediction_length, hour = False):
        #normalize features
        self.x_scaler = StandardScaler()
        self.features = self.x_scaler.fit_transform(l1_df.values)
        #dst scaler
        self.dst_scaler = StandardScaler() #
        self.dst = self.dst_scaler.fit_transform(dst_series.values.reshape(-1,1))
        #Kp scaler
        self.kp = kp_series.apply(map_kp_index_to_interval).values.reshape(-1,1)
        #other parameters
        self.sequence_length = sequence_length
        self.mode = hour
        self.pred_length = prediction_length #this will define how many hours later we want to train our model on 
    def __len__(self):
        return self.features.shape[0] - (self.sequence_length + self.pred_length) + 1
    def __getitem__(self, idx):
        feature = self.features[idx:idx+self.sequence_length, :]
        if self.mode:
            dst = self.dst[idx+self.sequence_length:idx+self.sequence_length+self.pred_length+1]
            kp = self.kp[hour_to_3_hour(idx+self.sequence_length):hour_to_3_hour(idx+self.sequence_length) + hour_to_3_hour(self.pred_length)+1]
        else:
            dst = self.dst[min_to_hour(idx+self.sequence_length):min_to_hour(idx+self.sequence_length)+self.pred_length]
            kp = self.kp[hour_to_3_hour(min_to_hour(idx+self.sequence_length)):hour_to_3_hour(min_to_hour(idx+self.sequence_length))+hour_to_3_hour(min_to_hour(self.pred_length))+1]
        feature = torch.tensor(feature, dtype=torch.float32)
        dst = torch.tensor(dst, dtype=torch.float32)
        kp = torch.tensor(kp, dtype=torch.float32)
        return feature, dst, kp
    
