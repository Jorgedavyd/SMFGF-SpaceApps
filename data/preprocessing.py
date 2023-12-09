from torchvision.datasets.utils import download_url, download_file_from_google_drive
from datetime import datetime, timedelta,date
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from datetime import date
import pandas as pd
import numpy as np
import csv
import zipfile
import tarfile
from urllib.request import urlopen
import glob
from astropy.time import Time
from sunpy.net import Fido, attrs as a
import astropy.units as u
import shutil
import spacepy.pycdf as pycdf
from math import sqrt

"""
ACE SPACECRAFT (ESA)
"""

def SIS_version(date, mode = '%Y%m%d'):
    date = datetime.strptime(date, mode)
    v5 = datetime.strptime('20141104', '%Y%m%d')
    v6 = datetime.strptime('20171019', '%Y%m%d')
    if date<v5:
        return 'v04'
    elif date<v6:
        return 'v05'
    else:
        return 'v06'

def EPAM_version(date, mode = '%Y%m%d'):
    date = datetime.strptime(date, mode)
    v5 = datetime.strptime('20150101', '%Y%m%d')
    if date<v5:
        return 'v04'
    else:
        return 'v05'

def MAG_version(date, mode = '%Y%m%d'):
    date = datetime.strptime(date, mode)
    v5 = datetime.strptime('20030328', '%Y%m%d')
    v6 = datetime.strptime('20120630', '%Y%m%d')
    v7 = datetime.strptime('20180130', '%Y%m%d')
    if date<v5:
        return 'v04'
    elif date<v6:
        return 'v05'
    elif date<v7:
        return 'v06'
    else:
        return 'v07'

def SWEPAM_version(date, mode = '%Y%m%d'):
    date = datetime.strptime(date, mode)
    v7 = datetime.strptime('20031030', '%Y%m%d')
    v8 = datetime.strptime('20050227', '%Y%m%d')
    v9 = datetime.strptime('20050325', '%Y%m%d')
    v10 = datetime.strptime('20061207', '%Y%m%d')
    v11 = datetime.strptime('20130101', '%Y%m%d')
    
    if date<v7:
        return 'v06'
    elif date<v8:
        return 'v07'
    elif date<v9:
        return 'v08'
    elif date<v10:
        return 'v09'
    elif date<v11:
        return 'v10'
    else:
        return 'v11'


## https://cdaweb.gsfc.nasa.gov/cgi-bin/eval1.cgi
class ACE:
    def __init__(self):
        pass
    def SIS(self, scrap_date):
        try:
            csv_file = './data/ACE/SIS/data.csv' #directories
            temp_root = './data/ACE/SIS/temp' 
            os.makedirs(temp_root) #create folder
            phy_obs = [
                'flux_He', 'flux_C', 'flux_N', 'flux_O', 'flux_Ne', 'flux_Mg',
                'flux_Si', 'flux_S', 'flux_Ar', 'flux_Ca', 'flux_Fe',
                'flux_Ni', 'cnt_He', 'cnt_C', 'cnt_N', 'cnt_O', 'cnt_Ne',
                'cnt_Mg', 'cnt_Si', 'cnt_S', 'cnt_Ar', 'cnt_Ca',
                'cnt_Fe', 'cnt_Ni'
            ]
            variables = ['datetime'] + [f'{name}_{i}' for name in phy_obs for i in range(1,9)]
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                version = SIS_version(date)
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/ace/sis/level_2_cdaweb/sis_h1/{date[:4]}/ac_h1_sis_{date}_{version}.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    data_columns.append(cdf_file[var][:])

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def MAG(self, scrap_date):
        try:
            csv_file = './data/ACE/MAG/data.csv' #directories
            temp_root = './data/ACE/MAG/temp' 
            os.makedirs(temp_root) #create folder
            phy_obs = ['Magnitude', 'BGSM', 'SC_pos_GSM', 'dBrms', 'BGSEc','SC_pos_GSE']
            variables = ['datetime'] + phy_obs
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                version = MAG_version(date)
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/ace/mag/level_2_cdaweb/mfi_h0/{date[:4]}/ac_h0_mfi_{date}_{version}.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    data_columns.append(cdf_file[var][:])

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def SWEPAM(self, scrap_date):
        csv_file = './data/ACE/SWEPAM/data.csv' #directories
        try:
            os.makedirs(temp_root) #create folder
            temp_root = './data/ACE/SWEPAM/temp' 
            phy_obs = ['Np', 'Vp','Tpr','alpha_ratio', 'V_GSE', 'V_GSM'] #variables#change
            variables = ['datetime'] + phy_obs
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                version = SWEPAM_version(date)
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/ace/swepam/level_2_cdaweb/swe_h0/{date[:4]}/ac_h0_swe_{date}_{version}.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)
                data_columns = []

                for var in phy_obs:
                    data_columns.append(cdf_file[var][:])

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def SWICS(self, scrap_date):
        csv_file = './data/ACE/SWICS/data.csv' #directories
        try:
            os.makedirs(temp_root) #create folder
            temp_root = './data/ACE/SWICS/temp' 
            phy_obs = ['nH', 'vH','vthH'] #variables#change
            variables = ['datetime'] + phy_obs
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/ace/swics/level_2_cdaweb/swi_h6/{date[:4]}/ac_h6_swi_{date}_v03.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)
                data_columns = []

                for var in phy_obs:
                    data_columns.append(cdf_file[var][:])

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def EPAM(self, scrap_date):
        csv_file = './data/ACE/EPAM/data.csv' #directories
        try:    
            temp_root = './data/ACE/EPAM/temp' 
            phy_obs = ['Epoch',
                        'P7',
                        'P8',
                        'DE1',
                        'DE2',
                        'DE3',
                        'DE4',
                        'W3',
                        'W4',
                        'W5',
                        'W6',
                        'W7',
                        'W8',
                        'E1p',
                        'E2p',
                        'E3p',
                        'E4p',
                        'FP5p',
                        'FP6p',
                        'FP7p',
                        'Z2',
                        'Z2A',
                        'Z3',
                        'Z4',
                        'P1p',
                        'P2p',
                        'P3p',
                        'P4p',
                        'P5p',
                        'P6p',
                        'P7p',
                        'P8p',
                        'E4',
                        'FP5',
                        'FP6',
                        'FP7'] #All available readings
            os.makedirs(temp_root) #create folder
            variables = ['datetime'] + phy_obs#variables#change
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                version = EPAM_version(date)
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/ace/epam/level_2_cdaweb/epm_h1/{date[:4]}/ac_h1_epm_{date}_{version}.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)
                data_columns = []

                for var in phy_obs:
                    data_columns.append(cdf_file[var][:])

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
            
"""
SOHO SPACECRAFT (ESA)
"""

class SOHO:
    def __init__(self):
        pass
    """
    SEM
    
    The Solar Electron and Proton Monitor (SEM) measures the energy 
    spectrum and composition of solar and galactic cosmic rays,
    as well as solar protons. An increase in solar proton flux can 
    be associated with solar flares and CMEs, which can, in turn, 
    influence geomagnetic activity.
    """

    def CELIAS_SEM(self, scrap_date):
        years = set(date[:4] for date in scrap_date)
        root = 'data/SOHO_L2/CELIAS_SEM_15sec_avg'
        csv_root = os.path.join(root, 'data.csv')

        try:
            url = 'https://soho.nascom.nasa.gov/data/EntireMissionBundles/CELIAS_SEM_15sec_avg.tar.gz'
            name = 'CELIAS_SEM_15sec_avg.tar.gz'
            os.makedirs(root, exist_ok=True)
            if 'data.csv' in os.listdir(root):
                raise FileExistsError
            download_url(url, root, name)
            
            with tarfile.open(os.path.join(root, name), 'r') as tar:
                tar.extractall(root)
            
            with open(csv_root, 'w') as csv_file:
                csv_file.write("Julian,F_Flux,C_Flux\n")

            for year in years:
                data_rows = []  
                year_folder = os.path.join(root, year)
                for day in sorted(os.listdir(year_folder)):
                    file_path = os.path.join(year_folder, day)
                    with open(file_path, 'r') as txt:
                        lines = txt.readlines()[46:]  # Skip first 46 lines
                        for line in lines:
                            data = [line.split()[0]] + line.split()[-2:] #julian,flux
                            data_rows.append(data)
                
                with open(csv_root, 'a') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    for row in data_rows:
                        csv_writer.writerow(row)

                shutil.rmtree(year_folder)
            
            celias_sem = pd.read_csv(csv_root)
            celias_sem['datetime'] = pd.to_datetime(celias_sem['Julian'], unit='D', origin = 'julian')
            celias_sem.set_index('datetime', drop=True,inplace=True)
            celias_sem.drop('Julian', axis = 1, inplace = True)
            celias_sem.to_csv(csv_root)
        except FileExistsError:
            pass

        celias_sem = pd.read_csv(csv_root, parse_dates=['datetime'], index_col='datetime')
        return celias_sem
    
    """CELIAS PROTON MONITOR"""
    """It has YY,MON,DY,DOY:HH:MM:SS,SPEED,Np,Vth,N/S,V_He"""
    def CELIAS_PM(self, scrap_date):
        csv_root = 'data/SOHO_L2/CELIAS_Proton_Monitor_5min/data.csv'
        years = set(date[:4] for date in scrap_date)
        month_map = {
            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
            'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
            'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
        }
        
        try:
            root = 'data/SOHO_L2/CELIAS_Proton_Monitor_5min'
            name = 'CELIAS_Proton_Monitor_5min.tar.gz'
            url = 'https://soho.nascom.nasa.gov/data/EntireMissionBundles/CELIAS_Proton_Monitor_5min.tar.gz'
            os.makedirs(root, exist_ok=True)
            if 'data.csv' in os.listdir(root):
                raise FileExistsError
            download_url(url, root, name)
            
            with tarfile.open(os.path.join(root, name), 'r') as tar:
                tar.extractall(root)

            with open(csv_root, 'w') as csv_file:
                csv_file.write("datetime,SPEED,Np,Vth,N/S,V_He\n")

            for year in years:
                data_rows = []
                filename = f'{year}_CELIAS_Proton_Monitor_5min'
                file_path = os.path.join(root, filename + '.zip')
                with zipfile.ZipFile(file_path, 'r') as archive:
                    with archive.open(filename+'.txt') as txt:
                        lines = txt.readlines() ####
                        for line in lines[29:]:
                            vector = [item.decode('utf-8') for item in line.split()]
                            data = [vector[0]+'-'+month_map[vector[1]]+'-'+vector[2]+' '+':'.join(vector[3].split(':')[1:])] + vector[4:-7] #ignores the position of SOHO
                            data_rows.append(data)
                with open(csv_root, 'a') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    for row in data_rows:
                        csv_writer.writerow(row)

                os.remove(file_path)
            
        except FileExistsError:
            pass

        df = pd.read_csv(csv_root, parse_dates=['datetime'], index_col='datetime', date_format='%y-%m-%d %H:%M:%S')

        return df
    
    def COSTEP_EPHIN(self, scrap_date):
        #getting dates
        years = sorted(list(set([date[:4] for date in scrap_date]))) ##YYYMMDD
        root = './data/SOHO_L2/COSTEP_EPHIN_5min'
        csv_root = os.path.join(root, 'data.csv')
        try:
            url = 'https://soho.nascom.nasa.gov/data/EntireMissionBundles/COSTEP_EPHIN_L3_l3i_5min-EntireMission-ByYear.tar.gz'
            name = 'COSTEP_EPHIN_L3_l3i_5min-EntireMission-ByYear.tar.gz'
            os.makedirs(root, exist_ok=True)
            if 'data.csv' in os.listdir(root):
                raise FileExistsError
            download_url(url, root, name)
            with tarfile.open(os.path.join(root, name), 'r') as tar:
                tar.extractall(root)
            
            columns = [
                'year', 'month', 'day', 'hour', 'minute',
                'int_p4', 'int_p8', 'int_p25', 'int_p41',
                'int_h4', 'int_h8', 'int_h25', 'int_h41'
            ]

            with open(csv_root, 'w') as csv_file:
                csv_file.writelines(','.join(columns) + '\n')

            for year in years:
                data_rows = []
                filename =os.path.join(root, '5min', f'{year}.l3i')
                with open(filename, 'r') as txt:
                    lines = txt.readlines()
                    for line in lines[3:]:
                        data = line.split()[:3] + line.split()[4:6] + line.split()[8:12] + line.split()[20:24]
                        data = [item for item in data]
                        data_rows.append(data)
                with open(csv_root, 'a') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    for row in data_rows:
                        csv_writer.writerow(row)
                os.remove(filename)
            shutil.rmtree(os.path.join(root, '5min'))

            df = pd.read_csv(csv_root)
            df['datetime'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
            df = df.drop(['year', 'month', 'day', 'hour', 'minute'], axis=1)  
            df.set_index('datetime', inplace=True, drop = True)
            df.to_csv(csv_root)
        except FileExistsError:
            pass
        costep_ephin = pd.read_csv(csv_root, parse_dates=['datetime'], index_col='datetime', date_format='%Y-%m-%d %H:%M:%S')
        return costep_ephin
    

## other utilities

def update_scrap_date(scrap_date, root): #ALWAYS USING THE SAME SCALE
    files = os.listdir(root)
    for idx, day in enumerate(scrap_date):
        for file in files:
            if day in file:
                del scrap_date[idx]
    return scrap_date


def interval_year(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")

    current_date = start_date
    year_list = []

    while current_date <= end_date:
        year_list.append(str(current_date.year))  # Append the year part of the date
        current_date += timedelta(days=365)  # Increment by one year (365 days)

    return year_list

def interval_time(start_date_str, end_date_str, mode = '%Y%m%d'):
    start_date = datetime.strptime(start_date_str, mode)
    end_date = datetime.strptime(end_date_str, mode)

    current_date = start_date
    date_list = []

    while current_date <= end_date:
        date_list.append(current_date.strftime(mode))
        current_date += timedelta(days=1)
    return date_list




def import_Dst(months = [str(date.today()).replace('-', '')[:6]]):
    os.makedirs('data/Dst_index', exist_ok = True)
    for month in months:
        if month+'.csv' in os.listdir('data/Dst_index'):
            continue
        # Define the URL from the kyoto Dst dataset
        if int(str(month)[:4])==int(date.today().year):
            url = f'https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{month}/index.html'
        elif 2017<=int(str(month)[:4])<=int(date.today().year)-1:
            url = f'https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{month}/index.html'
        else:
            url = f'https://wdc.kugi.kyoto-u.ac.jp/dst_final/{month}/index.html'
        response = requests.get(url)

        if response.status_code == 200:
            data = response.text

            soup = BeautifulSoup(data, 'html.parser')
            data = soup.find('pre', class_='data')
            with open('data/Dst_index/'+ url.split('/')[-2]+'.csv', 'w') as file:
                file.write('\n'.join(data.text.replace('\n\n', '\n').replace('\n ','\n').split('\n')[7:39]).replace('-', ' -').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        else:
            print('Unable to access the site')


def day_Dst(interval_time):
    data_list = []
    for day in interval_time:
        today_dst = pd.read_csv(f'data/Dst_index/{day[:6]}.csv',index_col = 0, header = None).T[int(day[6:])]
        for i,k in enumerate(today_dst):
            if isinstance(k, str): 
                today_dst[i+1] = float(today_dst[i+1])
            if np.abs(today_dst[i+1])>500:
                today_dst[i+1] = np.nan
        
        data_list.append(today_dst)
    series = pd.concat(data_list, axis = 0).reset_index(drop=True)
    series.name = 'Dst'
    return series

def day_Kp(interval_time):
    data_list = []
    kp = pd.read_csv(f'data/Kp_index/data.csv',index_col = 0, header = None).T
    for day in interval_time:
            try:
                today_kp = kp[day][0:8]
            except IndexError:
                continue
            for i,k in enumerate(today_kp):
                if isinstance(k, str): 
                    if np.abs(float(today_kp[i+1][0]))>9:
                        today_kp[i+1] = np.nan
                if isinstance(k, (int, float)):
                    if np.abs(today_kp[i+1])>9:
                        today_kp[i+1] = np.nan
            
            data_list.append(today_kp)
    series = pd.concat(data_list, axis = 0).reset_index(drop=True)
    series.name = 'Kp'
    return series

def import_targets(interval_time):
    kp = day_Kp(interval_time)
    dst = day_Dst(interval_time)
    return dst, kp