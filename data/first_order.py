import torch.nn as nn
from data.preprocessing import *
import torch
from sklearn.preprocessing import StandardScaler

class FirstOrderData(nn.Dataset):
    def __init__(self, list_scrap_dates, sequence_length): #hours
        self.sequence_length = pd.Timedelta(hours = sequence_length)
        # Create lists from data
        ##DSCOVR's Faraday Cup
        self.f1m_list = []
        ## DSCOVR's magnetometer
        self.m1m_list = []
        ## ACE's MAG
        self.ace_mag_list = []
        ## ACE's EPAM
        self.ace_epam_list = []
        ## ACE's SIS
        self.ace_sis_list = []
        ## ACE's SWICS
        self.ace_swics_list = []
        ## ACE's SWEPAM
        self.ace_swepam_list = []
        ## ACE's Proton monitor
        self.soho_pm_list = []
        ## SOHO's SEM
        self.soho_sem_list = []
        ## SOHO's COSTEP EPHIN
        self.soho_costep_list = []
        for scrap_date in list_scrap_dates:
            ## DSCOVR
            dscovr = DSCOVR()
            f1m, m1m = dscovr.MAGFC(scrap_date)
            self.f1m_list.append(f1m)
            self.m1m_list.append(m1m)
            ## ACE
            ace = ACE()
            self.ace_mag_list.append(ace.MAG(scrap_date))
            self.ace_epam_list.append(ace.EPAM(scrap_date))
            self.ace_sis_list.append(ace.SIS(scrap_date))
            self.ace_swics_list.append(ace.SWICS(scrap_date))
            self.ace_swepam_list.append(ace.SWEPAM(scrap_date))
            ## SOHO
            soho = SOHO()
            self.soho_pm_list.append(soho.CELIAS_PM(scrap_date))
            self.soho_costep_list.append(soho.COSTEP_EPHIN(scrap_date))
            self.soho_sem_list.append(soho.CELIAS_SEM(scrap_date))

        ## Data Norm
        self.f1m_scaler = StandardScaler().fit(pd.concat(self.f1m_list).values)
        self.m1m_scaler = StandardScaler().fit(pd.concat(self.m1m_list).values)
        self.ace_mag_scaler = StandardScaler().fit(pd.concat(self.ace_mag_list).values)
        self.ace_epam_scaler = StandardScaler().fit(pd.concat(self.ace_epam_list).values)
        self.ace_sis_scaler = StandardScaler().fit(pd.concat(self.ace_sis_list).values)
        self.ace_swics_scaler = StandardScaler().fit(pd.concat(self.ace_swics_list).values)
        self.ace_swepam_scaler = StandardScaler().fit(pd.concat(self.ace_swepam_list).values)
        self.soho_pm_scaler = StandardScaler().fit(pd.concat(self.soho_pm_list).values)
        self.soho_costep_scaler = StandardScaler().fit(pd.concat(self.soho_costep_list).values)
        self.soho_sem_scaler = StandardScaler().fit(pd.concat(self.soho_sem_list).values)
        
        #Mustering all datasets onto list
    def __len__(self):
        return sum([f1m.shape[0] for f1m in self.f1m_list]) - self.f1m_sequence_length*len(self.f1m_list) + len(self.f1m_list)
    def __getitem__(self, index):
        # Determine which dataset the sample belongs to
        dataset_idx = 0
        cumulative_length = len(self.f1m_list[0])
        while index >= cumulative_length:
            dataset_idx += 1
            cumulative_length += len(self.f1m_list[dataset_idx])

        # Adjust the index relative to the chosen dataset
        if dataset_idx > 0:
            index -= sum(len(self.datasets[i]) for i in range(dataset_idx))
        index = self.f1m_list[dataset_idx].index[index] 
        f1m = self.f1m_list[dataset_idx][index: index+self.sequence_length].values
        m1m = self.m1m_list[dataset_idx][index: index+self.sequence_length].values
        ace_mag = self.ace_mag_list[dataset_idx][index: index+self.sequence_length].values
        ace_epam = self.ace_epam_list[dataset_idx][index: index+self.sequence_length].values
        ace_sis = self.ace_sis_list[dataset_idx][index: index+self.sequence_length].values
        ace_swics = self.ace_swics_list[dataset_idx][index: index+self.sequence_length].values
        ace_swepam = self.ace_swepam_list[dataset_idx][index: index+self.sequence_length].values
        soho_pm = self.soho_pm_list[dataset_idx][index: index+self.sequence_length].values
        soho_costep = self.soho_costep_list[dataset_idx][index: index+self.sequence_length].values
        soho_sem = self.soho_sem_list[dataset_idx][index: index+self.sequence_length].values
        #Normalizing data
        f1m =  torch.from_numpy(self.f1m_scaler.transform(f1m), dtype = torch.float32)
        m1m =  torch.from_numpy(self.m1m_scaler.transform(m1m), dtype = torch.float32)
        ace_mag =  torch.from_numpy(self.ace_mag_scaler.transform(ace_mag), dtype = torch.float32)
        ace_epam =  torch.from_numpy(self.ace_epam_scaler.transform(ace_epam), dtype = torch.float32)
        ace_sis =  torch.from_numpy(self.ace_sis_scaler.transform(ace_sis), dtype = torch.float32)
        ace_swics =  torch.from_numpy(self.ace_swics_scaler.transform(ace_swics), dtype = torch.float32)
        ace_swepam =  torch.from_numpy(self.ace_swepam_scaler.transform(ace_swepam), dtype = torch.float32)
        soho_pm =  torch.from_numpy(self.soho_pm_scaler.transform(soho_pm), dtype = torch.float32)
        soho_costep =  torch.from_numpy(self.soho_costep_scaler.transform(soho_costep), dtype = torch.float32)
        soho_sem =  torch.from_numpy(self.soho_sem_scaler.transform(soho_sem), dtype = torch.float32)
        mag_head = [m1m, ace_mag]
        epa_head = [f1m, ace_epam, ace_swics, ace_swepam, soho_pm, soho_costep, soho_sem]
        isotropyc_head = [ace_sis]

        return mag_head, epa_head, isotropyc_head