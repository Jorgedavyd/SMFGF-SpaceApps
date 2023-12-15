from viresclient import SwarmRequest
import pandas as pd
from datetime import date, timedelta, datetime
from bs4 import BeautifulSoup
import numpy as np
import os
import requests

class SWARM:
    def MAG_x(self, scrap_date, sc = 'A'): #spacecrafts = ['A', 'B', 'C'] #scrap_date format YYYY-MM-DD
        try:
            csv_file_root = f'./data/SWARM/MAG{sc}/{scrap_date[0]}_{scrap_date[-1]}.csv'
            mag_x = pd.read_csv(csv_file_root, parse_dates = ['Timestamp'], index_col = 'Timestamp')
            mag_x.index = pd.to_datetime(mag_x.index)
            return mag_x
        except FileNotFoundError:
            request = SwarmRequest()
            # - See https://viresclient.readthedocs.io/en/latest/available_parameters.html
            request.set_collection(f"SW_OPER_MAG{sc}_LR_1B")
            request.set_products(
                measurements=[
                    'F',
                    'dF_Sun',
                    'B_VFM',
                    'dB_Sun',
                    ],
                auxiliaries=["Dst"],
            )
            # Fetch data from a given time interval
            # - Specify times as ISO-8601 strings or Python datetime
            data = request.get_between(
                start_time= scrap_date[0] + 'T00:00',
                end_time= scrap_date[-1] + 'T23:59'
            )
            df = data.as_dataframe()
            b_VFM = pd.DataFrame(df['B_VFM'].tolist(), columns=['B_VFM_1', 'B_VFM_2', 'B_VFM_3'], index = df.index)
            dB_Sun = pd.DataFrame(df['dB_Sun'].tolist(), columns=['dB_Sun_1', 'dB_Sun_2','dB_Sun_3'], index = df.index)
            df = pd.concat([df.drop(['B_VFM','dB_Sun','Spacecraft',], axis = 1), b_VFM, dB_Sun], axis = 1).resample('5T').mean(skipna = True)
            df.columns = ['Longitude', 'Dst','dF_Sun','F', 'Radius', 'Latitude', 'b_VFM_1', 'b_VFM_2', 'b_VFM_3', 'dB_Sun_1', 'dB_Sun_2','dB_Sun_3']
            os.makedirs(f'./data/SWARM/MAG{sc}', exist_ok = True)
            df.to_csv(csv_file_root)
            return df 
    def ION_x(self, scrap_date, sc = 'A'): #spacecrafts = ['A', 'B', 'C'] #scrap_date format YYYY-MM-DD
        try:
            csv_file_root = f'./data/SWARM/ion_plasma_{sc}/{scrap_date[0]}_{scrap_date[-1]}.csv'
            ion_x = pd.read_csv(csv_file_root, parse_dates = ['Timestamp'], index_col = 'Timestamp')
            return ion_x
        except FileNotFoundError:
            par_dict = {
                'Electric field instrument (Langmuir probe measurements at 2Hz)': [f"SW_OPER_EFI{sc}_LP_1B", ['Ne', 'Te', 'Vs','U_orbit']], #collection, measurements, 
                '16Hz cross-track ion flows': [f'SW_EXPT_EFI{sc}_TCT16', ['Ehx','Ehy','Ehz','Vicrx','Vicry','Vicrz']], 
                'Estimates of the ion temperatures': [f'SW_OPER_EFI{sc}TIE_2_',['Tn_msis', 'Ti_meas_drift']],
                '2Hz ion drift velocities and effective masses (SLIDEM project)': [f'SW_PREL_EFI{sc}IDM_2_', ['V_i','N_i', 'M_i_eff']]
            }
            df_list = []
            for parameters in par_dict.values():
                request = SwarmRequest()
                # - See https://viresclient.readthedocs.io/en/latest/available_parameters.html
                request.set_collection(parameters[0])
                request.set_products(
                    measurements=parameters[1]
                )
                # Fetch data from a given time interval
                # - Specify times as ISO-8601 strings or Python datetime
                data = request.get_between(
                    start_time= scrap_date[0] + 'T00:00',
                    end_time= scrap_date[-1] + 'T23:59',
                )
                df = data.as_dataframe().drop(['Longitude','Spacecraft','Radius','Latitude'], axis = 1)
                df.index = pd.to_datetime(df.index)
                df = df.resample('5T').mean(skipna = True) ##solve quality flags 
                df_list.append(df)
                del request
                del data
            os.makedirs(f'./data/SWARM/ion_plasma_{sc}', exist_ok = True)
            df = pd.concat(df_list, axis = 1)
            df.to_csv(csv_file_root)
            return df 

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


def interval_time(start_date_str, end_date_str, format = "%Y%m%d"):
    start_date = datetime.strptime(start_date_str, format)
    end_date = datetime.strptime(end_date_str, format)

    current_date = start_date
    date_list = []

    while current_date <= end_date:
        date_list.append(current_date.strftime(format))
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