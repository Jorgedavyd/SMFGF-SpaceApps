import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
def idx_to_hour(idx):
    return torch.floor(idx/60)

class RefinedTrainingDataset(Dataset):
    def __init__(self, l1_df, l2_df, target_df, sequence_length, prediction_length, hour = False):
        #l1 scaler
        self.x_scaler = StandardScaler()
        self.raw = self.x_scaler.fit_transform(l1_df.values)
        #l2 scaler
        self.x_hat_scaler = StandardScaler()
        self.pro = self.x_hat_scaler.fit_transform(l2_df.values)
        #output scaler
        self.y_scaler = StandardScaler()
        self.ground_truth = self.y_scaler.fit_transform(target_df.values)
        #other parameters
        self.sequence_length = sequence_length
        self.mode = hour
        self.pred_length = prediction_length #this will define how many hours later we want to train our model on 
    def __len__(self):
        return self.raw.shape[1] - (self.sequence_length + self.pred_length) + 1
    def __getitem__(self, idx):
        l1_sample = self.raw[idx:idx+self.sequence_length, :]
        l2_sample = self.pro[idx:idx+self.sequence_length, :]
        if self.mode:
            label = self.ground_truth[idx+self.sequence_length:idx+self.sequence_length+self.pred_length]
        else:
            label = self.ground_truth[idx_to_hour(idx+self.sequence_length):idx_to_hour(idx+self.sequence_length)+self.pred_length]
        
        return l1_sample, l2_sample, label
    
class NormalTrainingDataset(Dataset):
    def __init__(self, l1_df, target_df, sequence_length, prediction_length, hour = False):
        #normalize features
        self.x_scaler = StandardScaler()
        self.features = self.x_scaler.fit_transform(l1_df.values)
        #normalize target
        self.y_scaler =  StandardScaler()
        self.labels = self.y_scaler.fit_transform(target_df.values)
        #other parameters
        self.sequence_length = sequence_length
        self.mode = hour
        self.pred_length = prediction_length #this will define how many hours later we want to train our model on 
    def __len__(self):
        return self.features.shape[1] - (self.sequence_length + self.pred_length) + 1
    def __getitem__(self, idx):
        feature = self.features[idx:idx+self.sequence_length, :]
        if self.mode:
            label = self.labels[idx+self.sequence_length:idx+self.sequence_length+self.pred_length]
        else:
            label = self.labels[idx_to_hour(idx+self.sequence_length):idx_to_hour(idx+self.sequence_length)+self.pred_length]
        
        return feature, label
    
