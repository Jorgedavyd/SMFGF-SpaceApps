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

#format: YYYYMMDD000000
def import_train(scrap_date: list):
    with open('data/URLs.csv', 'r') as file:
        line = file.readlines()
        full_url = line[0].split(' ')
    url_list = []
    for url in full_url:
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

    ##uncompress and save
    for file in os.listdir('data/compressed'):
        output_file = os.path.join('data/uncompressed/',file[:-3]) 
        file = os.path.join('data/compressed/',file)
        if 'fc1' in file:
            fc1.append(output_file)
        else:
            mg1.append(output_file)
        with gzip.open(file, 'rb') as compressed_file:
            with open(output_file, 'wb') as decompressed_file:
                decompressed_file.write(compressed_file.read())
    return fc1, mg1

def faraday_preprocess(dataframes):
    
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

def concat_magnet(dataframes):
    
    data_list = []

    for nc_file in dataframes:
        dataset = xr.open_dataset(nc_file)

        df = dataset.to_dataframe()

        important_variables = ['bt','bx_gse', 'by_gse', 'bz_gse', 'theta_gse', 'phi_gse', 'bx_gsm','by_gsm', 'bz_gsm', 'theta_gsm', 'phi_gsm']

        magnetometer = df[important_variables]

        magnetometer = magnetometer.resample('1min').mean()

        data_list.append(magnetometer)
    return pd.concat(data_list)

def level_1_preprocessing():
    fc1, mg1 = gzip_to_nc()
    faraday = faraday_preprocess(fc1)
    magnetometer = concat_magnet(mg1)
    return pd.concat([faraday, magnetometer], axis =1)


def import_Dst(months = [str(date.today()).replace('-', '')[:6]]):
    for month in months:
        # Define the URL you want to download data from
        if int(str(month)[:4])==int(date.today().year):
            url = f'https://wdc.kugi.kyoto-u.ac.jp/dst_realtime/{month}/index.html'
        elif 2017<=int(str(month)[:4])<=int(date.today().year)-1:
            url = f'https://wdc.kugi.kyoto-u.ac.jp/dst_provisional/{month}/index.html'
        else:
            url = f'https://wdc.kugi.kyoto-u.ac.jp/dst_final/{month}/index.html'
        # Send an HTTP GET request to the URL
        response = requests.get(url)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the data (assuming it's CSV)
            data = response.text

            soup = BeautifulSoup(data, 'html.parser')
            data = soup.find('pre', class_='data')
            with open('data/Dst_index/'+ url.split('/')[-2]+'.csv', 'w') as file:
                file.write('\n'.join(data.text.replace('\n\n', '\n').replace('\n ','\n').split('\n')[7:39]).replace('-', ' -').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
        else:
            print('Unable to access the site')
#format: month: YYYYMM day: D for < 10 and DD for > 10.

def day_Dst(months = [str(date.today()).replace('-', '')[:6]], days = [date.today().day]):
    data_list = []
    for month in months:
        for day in days:
            today_dst = pd.read_csv(f'data/Dst_index/{month}.csv',index_col = 0, header = None).T[day]

            for i,k in enumerate(today_dst):
                if isinstance(k, str): 
                    today_dst[i+1] = float(today_dst[i+1])
                if np.abs(today_dst[i+1])>500:
                    today_dst[i+1] = np.nan
            
            data_list.append(today_dst)
    series = pd.concat(data_list, axis = 0).reset_index(drop=True)
    series.name = 'Dst'
    return series

def day_independent():
    url = 'https://www.ngdc.noaa.gov/dscovr/portal/#/download/'   
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the data (assuming it's CSV)
        data = response.text

        soup = BeautifulSoup(data, 'html.parser')
        data = soup.find('input', class_="form-control input-sm cursor-text")
        with open('data/Dst_index/'+ url.split('/')[-2]+'.csv', 'w') as file:
            file.write('\n'.join(data.text.replace('\n\n', '\n').replace('\n ','\n').split('\n')[7:39]).replace('-', ' -').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
    else:
        print('Unable to access the site')