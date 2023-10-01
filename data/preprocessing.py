import xarray as xr
import pandas as pd
import os
import gzip
import requests
import csv
from bs4 import BeautifulSoup
from datetime import date
import pandas as pd
import numpy as np
from torchvision.datasets.utils import download_url
from datetime import datetime, timedelta

#format: YYYYMMDD000000
def import_train(scrap_date: list):
    with open('data/URLs.csv', 'r') as file:
        lines = file.readlines()
    url_list = []
    for url in lines:
        for date in scrap_date:
            if date+'000000' in url:
                url_list.append(url)
    for url in url_list:
        root = 'data/compressed'
        filename = url.split('_')[1] + '_'+url.split('_')[3][1:-6]+'.nc.gz'
        download_url(url, root, filename)

def gzip_to_nc():
    #defining raw data dataframes
    fc1 = []
    mg1 = []
    f1m = []
    m1m = []

    ##uncompress and save
    for file in os.listdir('data/compressed'):
        output_file = os.path.join('data/uncompressed/',file[:-3]) 
        file = os.path.join('data/compressed/',file)
        if 'fc1' in file:
            fc1.append(output_file)
        elif 'f1m' in file:
            f1m.append(output_file)
        elif 'm1m' in file:
            m1m.append(output_file)
        else:
            mg1.append(output_file)
        with gzip.open(file, 'rb') as compressed_file:
            with open(output_file, 'wb') as decompressed_file:
                decompressed_file.write(compressed_file.read())
    return fc1, mg1, f1m, m1m

def l1_faraday_preprocess(dataframes):
    
    data_list = []
    
    #cleaning data

    for nc_file in dataframes: 

        dataset = xr.open_dataset(nc_file)

        df = dataset.to_dataframe()

        important_variables = ['proton_vx_gse', 'proton_vy_gse', 'proton_vz_gse', 'proton_vx_gsm', 'proton_vy_gsm', 'proton_vz_gsm', 'proton_speed', 'proton_density', 'proton_temperature']

        faraday_cup = df[important_variables]

        faraday_cup = faraday_cup.resample('1min').mean()

        data_list.append(faraday_cup)
    return pd.concat(data_list)

def l1_magnet_preprocess(dataframes):
    
    data_list = []

    for nc_file in dataframes:
        dataset = xr.open_dataset(nc_file)

        df = dataset.to_dataframe()

        important_variables = ['bt','bx_gse', 'by_gse', 'bz_gse', 'theta_gse', 'phi_gse', 'bx_gsm','by_gsm', 'bz_gsm', 'theta_gsm', 'phi_gsm']

        magnetometer = df[important_variables]

        magnetometer = magnetometer.resample('1min').mean()

        data_list.append(magnetometer)
    return pd.concat(data_list)

def l2_faraday_preprocess(dataframes):
    
    data_list = []
    
    #cleaning data

    for nc_file in dataframes: 

        dataset = xr.open_dataset(nc_file)

        df = dataset.to_dataframe()

        important_variables = ['proton_vx_gse', 'proton_vy_gse', 'proton_vz_gse', 'proton_vx_gsm', 'proton_vy_gsm', 'proton_vz_gsm', 'proton_speed', 'proton_density', 'proton_temperature']

        faraday_cup = df[important_variables]

        data_list.append(faraday_cup)
    return pd.concat(data_list)

def l2_magnet_preprocess(dataframes):
    
    data_list = []

    for nc_file in dataframes:
        dataset = xr.open_dataset(nc_file)

        df = dataset.to_dataframe()

        important_variables = ['bt','bx_gse', 'by_gse', 'bz_gse', 'theta_gse', 'phi_gse', 'bx_gsm','by_gsm', 'bz_gsm', 'theta_gsm', 'phi_gsm']

        magnetometer = df[important_variables]

        data_list.append(magnetometer)
    return pd.concat(data_list)

def preprocessing():
    fc1, mg1, f1m, m1m = gzip_to_nc()
    l1_faraday = l1_faraday_preprocess(fc1)
    l1_magnetometer = l1_magnet_preprocess(mg1)
    l2_faraday = l2_faraday_preprocess(f1m)
    l2_magnetometer = l2_magnet_preprocess(m1m)
    return pd.concat([l1_faraday, l1_magnetometer], axis =1), pd.concat([l2_faraday, l2_magnetometer], axis =1)

def import_Dst(months = [str(date.today()).replace('-', '')[:6]]):
    for month in months:
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


def interval_time(start_date_str, end_date_str):
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")

    current_date = start_date
    date_list = []

    while current_date <= end_date:
        date_list.append(current_date.strftime("%Y%m%d"))
        current_date += timedelta(days=1)

    return date_list
#format: month: YYYYMM day: D for < 10 and DD for > 10.

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
                    today_kp[i+1] = float(today_kp[i+1])
                if np.abs(today_kp[i+1])>9:
                    today_kp[i+1] = np.nan
            
            data_list.append(today_kp)
    series = pd.concat(data_list, axis = 0).reset_index(drop=True)
    series.name = 'Kp'
    return series

def day_Ap(interval_time):
    data_list = []
    ap = pd.read_csv(f'data/Kp_index/data.csv',index_col = 0, header = None).T
    for day in interval_time:
            try:
                today_kp = ap[day][9:17]
            except IndexError:
                continue
            for i,k in enumerate(today_kp):
                if isinstance(k, str): 
                    today_kp[i+1] = float(today_kp[i+1])
                if np.abs(today_kp[i+1])>9:
                    today_kp[i+1] = np.nan
            
            data_list.append(today_kp)
    series = pd.concat(data_list, axis = 0).reset_index(drop=True)
    series.name = 'ap'
    return series

def import_targets(interval_time):
    kp = day_Kp(interval_time)
    ap = day_Ap(interval_time)
    dst = day_Dst(interval_time)
    return dst, kp, ap