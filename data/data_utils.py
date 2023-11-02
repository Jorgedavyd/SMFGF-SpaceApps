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
    
def kp_to_gscale(kp_index):
    if kp_index < 4:
        return 0  # No Geomagnetic Storm
    elif kp_index < 5.67:
        return 1  # G1
    elif kp_index < 6.67:
        return 2  # G2
    elif kp_index < 7.67:
        return 3  # G3
    else: 
        return 4 #G4 or G5
    
def dst_to_gscale(dst_index):
    if dst_index >= -30:
        return 0  # No Geomagnetic Storm
    elif dst_index >= -50:
        return 1  # G1
    elif dst_index >= -100:
        return 2  # G2
    elif dst_index >= -200:
        return 3  # G3
    else:
        return 4 #G4 or G5

#We need the pred_length to be of size divisible by 3 if possible
class RefinedTrainingDataset(Dataset):
    def __init__(self, l1_df, l2_df, dst_series, kp_series, sequence_length, prediction_length, hour = False, time_step_ahead = 0, multiclass = True):
        self.time_ahead = time_step_ahead
        #other parameters
        self.sequence_length = sequence_length
        self.mode = hour
        self.pred_length = prediction_length #this will define how many hours later we want to train our model on 
        #l1 scaler
        self.x_scaler = StandardScaler()
        self.raw = self.x_scaler.fit_transform(l1_df.values)
        #l2 scaler
        self.x_hat_scaler = StandardScaler()
        self.pro = self.x_hat_scaler.fit_transform(l2_df.values)
        if multiclass:
            self.dst = dst_series.apply(dst_to_gscale).values.reshape(-1,1)
            self.kp = kp_series.apply(map_kp_index_to_interval).apply(dst_to_gscale).values.reshape(-1,1)
        else:
            self.dst_scaler = StandardScaler() #
            self.dst = self.dst_scaler.fit_transform(dst_series.values.reshape(-1,1))
            self.kp_scaler = StandardScaler() #
            self.kp = kp_series.apply(map_kp_index_to_interval).values.reshape(-1,1)
            self.kp = self.kp_scaler.fit_transform(self.kp)

    def __len__(self):
        if self.mode:
            return self.raw.shape[0] - (self.sequence_length + self.pred_length+ self.time_ahead) + 1
        else:
            return self.raw.shape[0] - (self.sequence_length + 60*self.pred_length+ self.time_ahead) + 1
            
    def __getitem__(self, idx):
        l1_sample = self.raw[idx:idx+self.sequence_length, :]
        l2_sample = self.pro[idx:idx+self.sequence_length, :]
        if self.mode:
            dst = self.dst[idx+self.sequence_length+ self.time_ahead:idx+self.sequence_length+self.pred_length+ self.time_ahead]
            kp = self.kp[hour_to_3_hour(idx+self.sequence_length)+ self.time_ahead:hour_to_3_hour(idx+self.sequence_length)+ self.time_ahead + hour_to_3_hour(self.pred_length)]
        else:
            dst = self.dst[min_to_hour(idx+self.sequence_length)+ self.time_ahead:min_to_hour(idx+self.sequence_length)+ self.time_ahead+self.pred_length]
            kp = self.kp[hour_to_3_hour(min_to_hour(idx+self.sequence_length))+ self.time_ahead:hour_to_3_hour(min_to_hour(idx+self.sequence_length))+ self.time_ahead+hour_to_3_hour(self.pred_length)]
        l1_sample = torch.tensor(l1_sample, dtype=torch.float32)
        l2_sample = torch.tensor(l2_sample, dtype=torch.float32)
        dst = torch.tensor(dst, dtype=torch.float32).squeeze(1)
        kp = torch.tensor(kp, dtype=torch.float32).squeeze(1)
        return l1_sample, l2_sample, dst, kp

class NormalTrainingDataset(Dataset):
    def __init__(self, l1_df, dst_series, kp_series, sequence_length, prediction_length, hour = False, time_step_ahead = 0, multiclass = True):
        self.time_ahead = time_step_ahead
        #normalize features
        self.x_scaler = StandardScaler()
        self.features = self.x_scaler.fit_transform(l1_df.values)
        if multiclass:
            self.dst = dst_series.apply(dst_to_gscale).values.reshape(-1,1)
            self.kp = kp_series.apply(map_kp_index_to_interval).apply(dst_to_gscale).values.reshape(-1,1)
        else:
            self.dst_scaler = StandardScaler() #
            self.dst = self.dst_scaler.fit_transform(dst_series.values.reshape(-1,1))
            self.kp_scaler = StandardScaler() #
            self.kp = kp_series.apply(map_kp_index_to_interval).values.reshape(-1,1)
            self.kp = self.kp_scaler.fit_transform(self.kp)
        #other parameters
        self.sequence_length = sequence_length
        self.mode = hour
        self.pred_length = prediction_length #this will define how many hours later we want to train our model on 
    def __len__(self):
        if self.mode:
            return self.features.shape[0] - (self.sequence_length + self.pred_length+ self.time_ahead) + 1
        else:
            return self.features.shape[0] - (self.sequence_length + 60*self.pred_length+ self.time_ahead) + 1
    def __getitem__(self, idx):
        feature = self.features[idx:idx+self.sequence_length, :]
        if self.mode:
            dst = self.dst[idx+self.sequence_length+ self.time_ahead:idx+self.sequence_length+self.pred_length+ self.time_ahead]
            kp = self.kp[hour_to_3_hour(idx+self.sequence_length)+ self.time_ahead:hour_to_3_hour(idx+self.sequence_length)+ self.time_ahead + hour_to_3_hour(self.pred_length)]
        else:
            dst = self.dst[min_to_hour(idx+self.sequence_length)+ self.time_ahead:min_to_hour(idx+self.sequence_length)+ self.time_ahead+self.pred_length]
            kp = self.kp[hour_to_3_hour(min_to_hour(idx+self.sequence_length))+ self.time_ahead:hour_to_3_hour(min_to_hour(idx+self.sequence_length))+ self.time_ahead+hour_to_3_hour(self.pred_length)]
        feature = torch.tensor(feature, dtype=torch.float32)
        dst = torch.tensor(dst, dtype=torch.float32).squeeze(1)
        kp = torch.tensor(kp, dtype=torch.float32).squeeze(1)
        return feature, dst, kp

dict_values = ['dst_kyoto', 'kp_gfz']
class MainToSingleTarget(Dataset):
    def __init__(self, l1_df, target, sequence_length, prediction_length, hour = False, sep = False, target_mode:str = 'dst_kyoto', l2_df = None, time_step_ahead = 0, dae = False, multiclass = False):
        self.sep = sep
        self.dae = dae
        self.time_ahead = time_step_ahead
        if l2_df is not None:
            self.encoder_forcing = True
        else:
            self.encoder_forcing = False
        if sep:
            self.fc, self.mg = l1_df
            self.fc_scaler = StandardScaler()
            self.fc = self.fc_scaler.fit_transform(self.fc.values)
            self.mg_scaler = StandardScaler()
            self.mg = self.mg_scaler.fit_transform(self.mg.values)
            if self.encoder_forcing:
                self.f1m, self.m1m = l2_df
                self.f1m = self.fc_scaler.transform(self.f1m.values)
                self.m1m = self.mg_scaler.transform(self.m1m.values)
        else:
            self.x_scaler = StandardScaler()
            self.l1_data = self.x_scaler.fit_transform(l1_df.values)
            if self.encoder_forcing:
                self.l2_data = self.x_scaler.transform(l2_df.values)
        if multiclass:
            if target_mode == 'kp_gfz':
                self.target = target.apply(map_kp_index_to_interval).apply(kp_to_gscale).values.reshape(-1,1)
            elif target_mode == 'dst_kyoto':
                self.target = target.apply(dst_to_gscale).values.reshape(-1,1)
        else:
            #target scaler
            self.y_scaler = StandardScaler() #
            if target_mode =='kp_gfz':
                target = target.apply(map_kp_index_to_interval)
            self.target = self.y_scaler.fit_transform(target.values.reshape(-1,1))
        #other parameters
        self.sequence_length = sequence_length
        self.mode = hour
        self.pred_length = prediction_length #this will define how many hours later we want to train our model on 
        self.sep = sep
        self.target_mode = target_mode
    def __len__(self):
        if self.sep:
            if self.mode:
                return self.fc.shape[0] - (self.sequence_length + self.pred_length + self.time_ahead) + 1
            else:
                return self.fc.shape[0] - (self.sequence_length + 60*self.pred_length + self.time_ahead) + 1
        else:
            if self.mode:
                return self.l1_data.shape[0] - (self.sequence_length + self.pred_length + self.time_ahead) + 1
            else:
                return self.l1_data.shape[0] - (self.sequence_length + 60*self.pred_length + self.time_ahead) + 1
    def __getitem__(self, idx):
        if self.dae:
            if self.encoder_forcing == False:
                return 'You have to give l2 dataframe'
            l1_data = self.l1_data[idx:idx+self.sequence_length, :]
            l2_data = self.l2_data[idx:idx+self.sequence_length, :]
            l1_data = torch.tensor(l1_data, dtype = torch.float32)
            l2_data = torch.tensor(l2_data, dtype = torch.float32)
            return l1_data, l2_data
        if self.mode:
            if self.target_mode == 'kp_gfz':
                target = self.target[hour_to_3_hour(idx+self.sequence_length)+ self.time_ahead:hour_to_3_hour(idx+self.sequence_length) + hour_to_3_hour(self.pred_length)+ self.time_ahead]
            elif self.target_mode == 'dst_kyoto':
                target = self.target[idx+self.sequence_length+ self.time_ahead:idx+self.sequence_length+self.pred_length+ self.time_ahead]
        else:
            if self.target_mode == 'kp_gfz':
                target = self.target[hour_to_3_hour(min_to_hour(idx+self.sequence_length))+ self.time_ahead:hour_to_3_hour(min_to_hour(idx+self.sequence_length))+hour_to_3_hour(self.pred_length)+ self.time_ahead]
            elif self.target_mode == 'dst_kyoto':
                target = self.target[min_to_hour(idx+self.sequence_length)+ self.time_ahead:min_to_hour(idx+self.sequence_length)+self.pred_length+ self.time_ahead]
        target = torch.tensor(target, dtype=torch.float32).squeeze(1)
        if self.sep:
            fc = self.fc[idx:idx+self.sequence_length, :]
            mg = self.mg[idx:idx+self.sequence_length, :]
            fc = torch.tensor(fc, dtype=torch.float32)
            mg = torch.tensor(mg, dtype=torch.float32)
            if self.encoder_forcing:
                f1m = self.f1m[idx:idx+self.sequence_length, :]
                m1m = self.m1m[idx:idx+self.sequence_length, :]
                f1m = torch.tensor(f1m, dtype=torch.float32)
                m1m = torch.tensor(m1m, dtype=torch.float32)
                return fc, mg, f1m, m1m, target
            return fc, mg, target
        else:
            l1_data = self.l1_data[idx:idx+self.sequence_length, :]
            l1_data = torch.tensor(l1_data, dtype=torch.float32)
            if self.encoder_forcing:
                l2_data = self.l2_data[idx: idx+self.sequence_length, :]
                l2_data = torch.tensor(l2_data, dtype = torch.float32)
                return l1_data, l2_data, target
            return l1_data, target
    


