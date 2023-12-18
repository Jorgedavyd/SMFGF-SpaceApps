import torch.nn as nn
from data.preprocessing import *
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from datetime import datetime


class LagrangeOne(Dataset):
    def __init__(self, input_seqs: list, l1_seq, step_size = timedelta(minutes = 5), output_seq = None, swarm_seq = timedelta(hours = 2), swarm_deviation = timedelta(minutes = 20)):
        self.l1_seq_len = time_to_step(l1_seq, step_size)
        self.input_seqs= input_seqs
        self.output = output_seq
        if output_seq is not None:
            self.output_deviation = time_to_step(swarm_deviation, step_size)
            self.swarm_seq_len = time_to_step(swarm_seq, step_size)
    def __len__(self):
        return self.input_seqs[0].shape[0] - self.l1_seq_len + 1
    def __getitem__(self, idx):
        input_seqs = [input_seqs.values[idx: idx+self.l1_seq_len, :] for input_seqs in self.input_seqs]
        if self.output is not None:
            output_init = idx+self.l1_seq_len+self.output_deviation
            output = self.output.values[output_init: output_init + self.swarm_seq_len]
            output = output.reshape(-1,1) if len(output.shape) == 1 else output 
            return input_seqs, output
        else:
            return input_seqs

class Swarm2Dst(Dataset):
    def __init__(self, input_seqs: list, output_seq, seq_len, step_size = timedelta(minutes = 5)):
        self.seq_len = time_to_step(seq_len, step_size)
        self.input_seqs= input_seqs
        self.output_seqs = output_seq
    def __len__(self):
        return self.input_seqs[0].shape[0] - self.seq_len + 1
    def __getitem__(self, idx):
        input_seqs = [input_seq.values[idx: idx+self.seq_len, :] for input_seq in self.input_seqs]
        output_seq = self.output_seqs.values.reshape(-1,1)[idx:idx+self.seq_len, :]
        
        return input_seqs, output_seq

def time_shift(scrap_date, dev, sl):
    init_date = datetime.strptime(scrap_date[0],'%Y-%m-%d')
    end_date = datetime.strptime(scrap_date[-1],'%Y-%m-%d') + timedelta(days = 1) + dev + sl
    return [init_date, end_date]

def time_to_step(time, step_size=timedelta(minutes=5)):
    return int(time.total_seconds() / step_size.total_seconds())

def dst_time_shift(scrap_date, dev, sl):
    init_date = datetime.strptime(scrap_date[0], '%Y-%m-%d')
    end_date = datetime.strptime(scrap_date[-1], '%Y-%m-%d') + timedelta(days = 1) + dev + sl
    init_date = datetime.strftime(init_date, '%Y%m%d')
    end_date = datetime.strftime(end_date, '%Y%m%d')
    scrap_date = interval_time(init_date,end_date, format = '%Y%m%d')
    return scrap_date

class FirstStageModeling(Dataset):
    def __init__(self, scrap_date_list, swarm_sc = 'A', swarm_sl = timedelta(hours = 2), l1_sl=timedelta(days = 2), step_size = timedelta(minutes = 5), swarm_dev = timedelta(minutes = 30)):
        """
        L1 to SWARM

        This method targets SWARM sequences, we have these parameters:
        
        scrap_date_list: List of date intervals where you want to train your model. #lets to better dataset homogeneization
        swarm_sl: SWARM's sequence length.
        l1_sl: L1 Lagrange satellites sequence length.

        dscovr: DSCOVR satellite object
        swarm: SWARM satellite object
        resample_method: Time step size
        datasets: List of build-in SequentialDataset objects
        inputs_temp: Temp dataset for normalization
        output_temp: Temp dataset for normalization

        
        resample(resample_method).mean() is set in order to organize the data on identical
        time intervals
        
        """
        resample_method = '5T' #change for different step_size, 
        self.l1_sl = l1_sl
        self.swarm_sl = swarm_sl
        dscovr = DSCOVR()
        ace = ACE()
        swarm = SWARM()
        soho = SOHO()
        wind = WIND()
        ##DSCOVR scrap_date list
        l1_scrap = [interval_time(x,y, format = '%Y-%m-%d') for x,y in scrap_date_list]
        ## temp dataset for scalers
        inputs_temp = []
        output_temp = []
        self.datasets = [] # save on Sequential Dataset method
        for scrap_date in l1_scrap:
                
                input_seqs = [
                    *[tool.resample(resample_method).mean() for tool in dscovr.MAGFC(scrap_date)],## add here satellite by satellite
                    ace.EPAM(scrap_date),
                    ace.MAG(scrap_date),
                    ace.SWEPAM(scrap_date),
                    ace.SIS(scrap_date),
                    soho.CELIAS_PM(scrap_date),
                    soho.CELIAS_SEM(scrap_date),
                    soho.COSTEP_EPHIN(scrap_date),
                    wind.MAG(scrap_date),
                    wind.SMS(scrap_date),
                    wind.TDP_PLSP(scrap_date),
                    wind.TDP_PM(scrap_date),
                    wind.SWE_electron_moments(scrap_date),
                    wind.SWE_alpha_proton(scrap_date),
                    
                ] #creating the list of satellite data
                
                ##SWARM
                swarm_scrap = time_shift(scrap_date, swarm_dev, swarm_sl)
                output = (swarm.MAG_x(swarm_scrap, swarm_sc).resample(resample_method).mean().drop(['Dst', 'Longitude', 'Dst', 'Radius', 'Latitude'], axis = 1))

                self.datasets.append(LagrangeOne(input_seqs, l1_sl, step_size, output, swarm_sl, swarm_dev))

                inputs_temp.append(input_seqs)
                output_temp.append(output)
        self.input_scalers = []
        for i in range(len(input_seqs)):
            temp_input = [input_seqs[i] for input_seqs in inputs_temp]
            self.input_scalers.append(StandardScaler().fit(pd.concat(temp_input, axis = 0).values))

        self.output_scaler = StandardScaler().fit(pd.concat(output_temp, axis = 0).values)
    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)
    def __getitem__(self, index):
        # Determine which dataset the sample belongs to
        dataset_idx = 0
        cumulative_length = len(self.datasets[0])
        while index >= cumulative_length:
            dataset_idx += 1
            cumulative_length += len(self.datasets[dataset_idx])

        # Adjust the index relative to the chosen dataset
        if dataset_idx > 0:
            index -= sum(len(self.datasets[i]) for i in range(dataset_idx))
        
        input_seqs, output = self.datasets[dataset_idx][index]
        
        transformed_inputs = []
        for scaler, data in zip(self.input_scalers, input_seqs):
            transformed_inputs.append(torch.from_numpy(scaler.transform(data)).to(torch.float32))

        del input_seqs

        transformed_output = torch.from_numpy(self.output_scaler.transform(output)).to(torch.float32)

        del output
        
        return *transformed_inputs, transformed_output

class SecondStageModeling(Dataset):
    def __init__(self, scrap_date_list, seq_len = timedelta(hours = 2), step_size = timedelta(minutes = 5), dev = timedelta(minutes = 20)):
        """
        SWARM to DST

        This method targets DST sequences, we have these parameters:
        
        scrap_date_list: List of date intervals where you want to train your model. #lets to better dataset homogeneization
        seq_len: Sequence length.

        swarm: SWARM satellite object
        resample_method: Time step size
        datasets: List of build-in SequentialDataset objects

        resample(resample_method).mean() is set in order to organize the data on identical
        time intervals
        
        """
        resample_method = '5T' #change for different step_size, 
        swarm = SWARM()
        scrap_list = [interval_time(x,y, format = '%Y-%m-%d') for x,y in scrap_date_list]
        ## temp dataset for scalers
        inputs_temp = []
        output_temp = []
        self.datasets = [] # save on Sequential Dataset method
        swarm_sc =[
            'A',
            'B',
            'C'
        ]
        for scrap_date in scrap_list:
                swarm_scrap = time_shift(scrap_date, dev, seq_len)
                input_seqs = [swarm.MAG_x(swarm_scrap, letter).resample(resample_method).mean() for letter in swarm_sc]
                
                dst_scrap = dst_time_shift(scrap_date, dev, seq_len)
                output = Dst(dst_scrap)

                inputs_temp.append(input_seqs)
                output_temp.append(output)

                self.datasets.append(Swarm2Dst(input_seqs, output, seq_len, step_size))
            
        self.input_scalers = []
        for i in range(len(input_seqs)):
            temp_input = [input_seqs[i] for input_seqs in inputs_temp]
            self.input_scalers.append(StandardScaler().fit(pd.concat(temp_input, axis = 0).values))

        self.output_scaler = StandardScaler().fit(pd.concat(output_temp, axis = 0).values.reshape(-1,1))
    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)
    def __getitem__(self, index):
        # Determine which dataset the sample belongs to
        dataset_idx = 0
        cumulative_length = len(self.datasets[0])
        while index >= cumulative_length:
            dataset_idx += 1
            cumulative_length += len(self.datasets[dataset_idx])

        # Adjust the index relative to the chosen dataset
        if dataset_idx > 0:
            index -= sum(len(self.datasets[i]) for i in range(dataset_idx))
        
        input_seqs, output = self.datasets[dataset_idx][index]
        
        transformed_inputs = []
        for scaler, data in zip(self.input_scalers, input_seqs):
            transformed_inputs.append(torch.from_numpy(scaler.transform(data)).to(torch.float32))

        del input_seqs

        transformed_output = torch.from_numpy(self.output_scaler.transform(output)).to(torch.float32)

        del output
        
        return *transformed_inputs, transformed_output

class FirstOrderModel(Dataset):
    def __init__(self, scrap_date_list, dst_sl = timedelta(hours = 2), l1_sl=timedelta(days = 2), step_size = timedelta(minutes = 5), dev = timedelta(minutes = 30)):
        """
        SWARM to DST

        This method targets DST sequences, we have these parameters:
        
        scrap_date_list: List of date intervals where you want to train your model. #lets to better dataset homogeneization
        seq_len: Sequence length.

        swarm: SWARM satellite object
        resample_method: Time step size
        datasets: List of build-in SequentialDataset objects

        resample(resample_method).mean() is set in order to organize the data on identical
        time intervals
        
        """
        resample_method = '5T' #change for different step_size, 
        scrap_list = [interval_time(x,y, format = '%Y-%m-%d') for x,y in scrap_date_list]
        ## temp dataset for scalers
        inputs_temp = []
        output_temp = []
        self.datasets = [] # save on Sequential Dataset method
        dscovr = DSCOVR()
        for scrap_date in scrap_list:
                
                input_seqs = [
                    *[tool.resample(resample_method).mean() for tool in dscovr.MAGFC(scrap_date)]## add here satellite by satellite
                ]
                dst_scrap = dst_time_shift(scrap_date, dev, dst_sl)
                output = Dst(dst_scrap)

                inputs_temp.append(input_seqs)
                output_temp.append(output)

                self.datasets.append(LagrangeOne(input_seqs, l1_sl, step_size, output, dst_sl, dev))
            
        self.input_scalers = []
        for i in range(len(input_seqs)):
            temp_input = [input_seqs[i] for input_seqs in inputs_temp]
            self.input_scalers.append(StandardScaler().fit(pd.concat(temp_input, axis = 0).values))

        self.output_scaler = StandardScaler().fit(pd.concat(output_temp, axis = 0).values.reshape(-1,1))
    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)
    def __getitem__(self, index):
        # Determine which dataset the sample belongs to
        dataset_idx = 0
        cumulative_length = len(self.datasets[0])
        while index >= cumulative_length:
            dataset_idx += 1
            cumulative_length += len(self.datasets[dataset_idx])

        # Adjust the index relative to the chosen dataset
        if dataset_idx > 0:
            index -= sum(len(self.datasets[i]) for i in range(dataset_idx))
        
        input_seqs, output = self.datasets[dataset_idx][index]
        
        transformed_inputs = []
        for scaler, data in zip(self.input_scalers, input_seqs):
            transformed_inputs.append(torch.from_numpy(scaler.transform(data)).to(torch.float32))

        del input_seqs

        transformed_output = torch.from_numpy(self.output_scaler.transform(output)).to(torch.float32)

        del output
        
        return *transformed_inputs, transformed_output

class DaeDataset(Dataset):
    def __init__(self, scrap_date_list, seq_len, step_size = timedelta(minutes = 5)):
        scrap_date_list = [interval_time(x,y, format = '%Y-%m-%d') for x,y in scrap_date_list]
        resample_method = '5T'
        dscovr = DSCOVR()
        self.datasets = []
        temp_dataset = []
        for scrap_date in scrap_date_list:
            fc1, mg1, f1m, m1m = dscovr.MAGFC(scrap_date, level = 'both')
            fc1 = fc1.resample(resample_method).mean()
            mg1 = mg1.resample(resample_method).mean()
            f1m = f1m.resample(resample_method).mean()
            m1m = m1m.resample(resample_method).mean()

            dataset = [fc1, mg1, f1m, m1m]
            self.datasets.append(LagrangeOne(dataset, seq_len, step_size))
            temp_dataset.append(dataset)
        
        self.input_scalers = []
        for i in range(len(dataset)):
            temp_input = [input_seqs[i] for input_seqs in temp_dataset]
            self.input_scalers.append(StandardScaler().fit(pd.concat(temp_input, axis = 0).values))
        del temp_dataset
    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)
    def __getitem__(self, index):
        # Determine which dataset the sample belongs to
        dataset_idx = 0
        cumulative_length = len(self.datasets[0])
        while index >= cumulative_length:
            dataset_idx += 1
            cumulative_length += len(self.datasets[dataset_idx])

        # Adjust the index relative to the chosen dataset
        if dataset_idx > 0:
            index -= sum(len(self.datasets[i]) for i in range(dataset_idx))
        
        input_seqs = self.datasets[dataset_idx][index]
        
        transformed_inputs = []
        for scaler, data in zip(self.input_scalers, input_seqs):
            transformed_inputs.append(torch.from_numpy(scaler.transform(data)).to(torch.float32))

        return transformed_inputs,

