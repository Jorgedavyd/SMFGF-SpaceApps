import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np

def min_to_hour(idx):
    return int(np.floor(idx/60))
def hour_to_3_hour(idx):
    return int(np.floor(idx/3))

def map_kp_index_to_interval(kp_index):
    kp_mapping = {
        '0': 0.00,
        '0+': 0.33,
        '1-': 0.66,
        '1': 1.00,
        '1+': 1.33,
        '2-': 1.66,
        '2': 2.00,
        '2+': 2.33,
        '3-': 2.66,
        '3': 3.00,
        '3+': 3.33,
        '4-': 3.66,
        '4': 4.00,
        '4+': 4.33,
        '5-': 4.66,
        '5': 5.00,
        '5+': 5.33,
        '6-': 5.66,
        '6': 6.00,
        '6+': 6.33,
        '7-': 6.66,
        '7': 7.00,
        '7+': 7.33,
        '8-': 7.66,
        '8': 8.00,
        '8+': 8.33,
        '9-': 8.66,
        '9': 9.00,
        '9+': 9.33,
    }

    return kp_mapping[kp_index]
    


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
            kp = self.kp[hour_to_3_hour(min_to_hour(idx+self.sequence_length)):hour_to_3_hour(min_to_hour(idx+self.sequence_length))+hour_to_3_hour(self.pred_length)]
        l1_sample = torch.tensor(l1_sample, dtype=torch.float32)
        l2_sample = torch.tensor(l2_sample, dtype=torch.float32)
        dst = torch.tensor(dst, dtype=torch.float32).squeeze(1)
        kp = torch.tensor(kp, dtype=torch.float32).squeeze(1)
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
            dst = self.dst[idx+self.sequence_length:idx+self.sequence_length+self.pred_length]
            kp = self.kp[hour_to_3_hour(idx+self.sequence_length):hour_to_3_hour(idx+self.sequence_length) + hour_to_3_hour(self.pred_length)]
        else:
            dst = self.dst[min_to_hour(idx+self.sequence_length):min_to_hour(idx+self.sequence_length)+self.pred_length]
            kp = self.kp[hour_to_3_hour(min_to_hour(idx+self.sequence_length)):hour_to_3_hour(min_to_hour(idx+self.sequence_length))+hour_to_3_hour(self.pred_length)]
        feature = torch.tensor(feature, dtype=torch.float32)
        dst = torch.tensor(dst, dtype=torch.float32).squeeze(1)
        kp = torch.tensor(kp, dtype=torch.float32).squeeze(1)
        return feature, dst, kp
    
