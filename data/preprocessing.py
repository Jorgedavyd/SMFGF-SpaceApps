from torchvision.datasets.utils import download_url
from datetime import datetime, timedelta,date
import xarray as xr
import pandas as pd
import os
import gzip
import requests
from bs4 import BeautifulSoup
from datetime import date
import pandas as pd
import numpy as np
import io 
import csv
import zipfile
import tarfile

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
        #getting dates
        csv_root = 'data/SOHO/SEM.csv' #define the root
        years = list(set([date[:4] for date in scrap_date]))
        try:
            #getting the csv
            root = 'data/SOHO/'
            os.makedirs(root)
            name = 'CELIAS_Proton_Monitor_5min.tar.gz'
            url = 'https://soho.nascom.nasa.gov/data/EntireMissionBundles/CELIAS_Proton_Monitor_5min.tar.gz'
            download_url(url, root, name)
            with tarfile.open(os.path.join(root, name), 'r') as tar:
                tar.extractall(root)
            
            # Create a CSV output buffer
            output_buffer = io.StringIO()
            csv_writer = csv.writer(output_buffer)

            # Write the CSV header
            csv_header = [
                "YY", "MON", "DY", "DOY:HH:MM:SS", "SPEED", "Np", "Vth", "N/S", "V_He"
                ]
            csv_writer.writerow(csv_header)
            for year in years:
                file = f'data/SOHO/{year}_CELIAS_Proton_Monitor_5min.zip'
                with zipfile.ZipFile(file, 'r') as archive:
                    with archive.open(f'{year}_CELIAS_Proton_Monitor_5min.txt') as txt:
                        lines = txt.readlines()
                        for line in lines[29:]:
                            data = line.split()[:-7] #ignores the position of SOHO
                            data = [item.decode('utf-8') for item in data]
                            csv_writer.writerow(data)
                os.remove(file)

            csv_content = output_buffer.getvalue()

            output_buffer.close()
            with open(csv_root, "w", newline="") as csv_file:
                csv_file.write(csv_content)        
        except FileExistsError:
            pass
        celias_proton_monitor = pd.read_csv(csv_root)
        return celias_proton_monitor
    
    """CELIAS PROTON MONITOR"""
    """It has YY,MON,DY,DOY:HH:MM:SS,SPEED,Np,Vth,N/S,V_He"""
    def CELIAS_Proton_Monitor(self, scrap_date):
        #getting dates
        csv_root = 'data/SOHO/CELIAS_proton.csv' #define the root
        years = list(set([date[:4] for date in scrap_date]))
        try:
            #getting the csv
            root = 'data/SOHO/'
            os.makedirs(root)
            name = 'CELIAS_Proton_Monitor_5min.tar.gz'
            url = 'https://soho.nascom.nasa.gov/data/EntireMissionBundles/CELIAS_Proton_Monitor_5min.tar.gz'
            download_url(url, root, name)
            with tarfile.open(os.path.join(root, name), 'r') as tar:
                tar.extractall(root)
            
            # Create a CSV output buffer
            output_buffer = io.StringIO()
            csv_writer = csv.writer(output_buffer)

            # Write the CSV header
            csv_header = [
                "YY", "MON", "DY", "DOY:HH:MM:SS", "SPEED", "Np", "Vth", "N/S", "V_He"
                ]
            csv_writer.writerow(csv_header)
            for year in years:
                file = f'data/SOHO/{year}_CELIAS_Proton_Monitor_5min.zip'
                with zipfile.ZipFile(file, 'r') as archive:
                    with archive.open(f'{year}_CELIAS_Proton_Monitor_5min.txt') as txt:
                        lines = txt.readlines()
                        for line in lines[29:]:
                            data = line.split()[:-7] #ignores the position of SOHO
                            data = [item.decode('utf-8') for item in data]
                            csv_writer.writerow(data)
                os.remove(file)
                
            csv_content = output_buffer.getvalue()

            output_buffer.close()
            with open(csv_root, "w", newline="") as csv_file:
                csv_file.write(csv_content)        
        except FileExistsError:
            pass
        df = pd.read_csv(csv_root)
        df[['DOY', 'HH', 'MM', 'SS']] = df['DOY:HH:MM:SS'].str.split(':', expand=True)

        df['datetime'] = pd.to_datetime(
            '20' + df['YY'].apply(str) + df['MON'] + df['DY'].apply(str) + ' ' +
            df['DOY'] + ' ' + df['HH'] + ':' +
            df['MM'] + ':' + df['SS'],
            format='%Y%b%d %j %H:%M:%S')
        df.set_index(df['datetime']).drop(['YY', 'MON', 'DY', 'DOY', 'DOY:HH:MM:SS', 'HH', 'MM', 'SS', 'datetime'], axis =1)
        return df
    def CELIAS(self, scrap_date):
        proton_monitor = self.CELIAS_Proton_Monitor(scrap_date)
        sem = self.CELIAS_SEM(scrap_date)
        df = pd.concat([proton_monitor, sem], axis = 1)
        return df
    """LASCO"""
    """
    LASCO is designed to observe the solar corona by creating artificial 
    eclipses. It can help detect and track the expansion of coronal mass ejections (CMEs), which, 
    as mentioned earlier, can lead to geomagnetic storms when they reach Earth.
    """
    def LASCO(self, scrap_date):
        ...
    def UVCS(self, scrap_date):
        ...
    def ERNE(self, scrap_date):
        ...
    def EIT(self, scrap_date):
        ...
    def SUMER(self, scrap_date):
        ...
    def COSTEP(self, scrap_date):
        ...
    def all(self, scrap_date):
        lasco = self.LASCO() #imagenes
        uvcs = self.UVCS()
        eit = self.EIT() #imagenes
        sumer = self.SUMER()#imagenes creo
        costep = self.COSTEP()
        celias = self.CELIAS()
        df = pd.concat([lasco, uvcs, eit, sumer, costep, celias], axis = 1)
        return df


def interval_years(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")

    current_date = start_date
    year_list = []

    while current_date <= end_date:
        year_list.append(str(current_date.year))  # Append the year part of the date
        current_date += timedelta(days=365)  # Increment by one year (365 days)

    return year_list

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