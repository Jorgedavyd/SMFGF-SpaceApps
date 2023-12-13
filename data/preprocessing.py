from torchvision.datasets.utils import download_url
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
import astropy.units as u
import shutil
import spacepy.pycdf as pycdf
import xarray as xr
import gzip
import shutil

def WIND_MAG_version(date, mode = '%Y%m%d'):
    date = datetime.strptime(date, mode)
    v4 = datetime.strptime('20230101', '%Y%m%d')
    v3 = datetime.strptime('20231121', '%Y%m%d')
    if date<v4:
        return 'v05'
    elif date<v3:
        return 'v04'
    else:
        return 'v03'
def WIND_SWE_version(date, mode = '%Y%m%d'):
    date = datetime.strptime(date, mode)
    v4 = datetime.strptime('20230101', '%Y%m%d')
    v3 = datetime.strptime('20231121', '%Y%m%d')
    if date<v4:
        return 'v05'
    elif date<v3:
        return 'v04'
    else:
        return 'v03'
def TDP_PM_version(date, mode = '%Y%m%d'):
    date = datetime.strptime(date, mode)
    v4 = datetime.strptime('20110114', '%Y%m%d')
    v5 = datetime.strptime('20111230', '%Y%m%d')
    if date<v4:
        return 'v03'
    elif date<v5:
        return 'v04'
    else:
        return 'v05'


"""
WIND Spacecraft
"""

class WIND:
    def MAG(self, scrap_date):
        try:
            csv_file = './data/WIND/MAG/data.csv' #directories
            temp_root = './data/WIND/MAG/temp' 
            os.makedirs(temp_root) #create folder
            phy_obs = ['BF1','BGSE','BGSM']
            variables = ['datetime', 'BF1'] + [f'{name}_{i}' for name in phy_obs[1:3] for i in range(1,4)]
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                version = WIND_MAG_version(date)
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/mfi/mfi_h0/{date[:4]}/wi_h0_mfi_{date}_{version}.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:].squeeze(1)]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def SWE_alpha_proton(self, scrap_date): #includes spacecraft position
        try:
            csv_file = './data/WIND/SWE/alpha_proton/data.csv' #directories
            temp_root = './data/WIND/SWE/temp' 
            os.makedirs(temp_root) #create folder
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = ['Proton_V_nonlin', 'Proton_VX_nonlin', 'Proton_VY_nonlin', 'Proton_VZ_nonlin', 'Proton_Np_nonlin', 'Alpha_V_nonlin', 'Alpha_VX_nonlin',
                         'Alpha_VY_nonlin', 'Alpha_VZ_nonlin', 'Alpha_Na_nonlin', 'xgse', 'ygse','zgse']
            variables = ['datetime'] + ['Vp', 'Vpx', 'Vpy', 'Vpz', 'Np', 'Va', 'Vax', 'Vay', 'Vaz', 'Na', 'xgse', 'ygse','zgse']
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/swe/swe_h1/{date[:4]}/wi_h1_swe_{date}_v01.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

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
    def SWE_electron_angle(self, scrap_date):
        try:
            csv_file = './data/WIND/SWE/electron_angle/data.csv' #directories
            temp_root = './data/WIND/SWE/temp' 
            os.makedirs(temp_root) #create folder
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = ['f_pitch_SPA','Ve'] ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_h3_swe_00000000_v01.skt
            variables = ['datetime'] + [f'f_pitch_SPA_{i}' for i in range(13)] + [f'Ve_{i}' for i in range(13)]
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/swe/swe_h3/{date[:4]}/wi_h3_swe_{date}_v01.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

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
    def SWE_electron_moments(self, scrap_date):
        try:
            csv_file = './data/WIND/SWE/electron_moments/data.csv' #directories
            temp_root = './data/WIND/SWE/temp' 
            os.makedirs(temp_root) #create folder
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = ['N_elec','TcElec', 'U_eGSE', 'P_eGSE', 'W_elec', 'Te_pal'] ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_h5_swe_00000000_v01.skt
            variables = ['datetime'] + phy_obs[:2] + [phy_obs[2] + f'_{i}' for i in range(1,4)]+ [phy_obs[3] + f'_{i}' for i in range(1,7)] + phy_obs[4:]
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/swe/swe_h5/{date[:4]}/wi_h5_swe_{date}_v01.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

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
    def TDP_PM(self, scrap_date):
        try:
            csv_file = './data/WIND/TDP/PM/data.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root) #create folder
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = ['P_VELS', 'P_TEMP','P_DENS','A_VELS','A_TEMP','A_DENS'] ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_h5_swe_00000000_v01.skt
            variables = ['datetime', 'Vpx','Vpy','Vpz', 'Tp','Np','Vax', 'Vay', 'Vaz','Ta','Na'] #GSE
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                version = TDP_PM_version(date)
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_pm/{date[:4]}/wi_pm_3dp_{date}_{version}.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

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
    def TDP_PLSP(self, scrap_date):
        try:
            csv_file = './data/WIND/TDP/PLSP/data.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9])
            phy_obs = ['FLUX', 'ENERGY', 'MOM.P.VTHERMAL', 'MOM.P.FLUX','MOM.P.PTENS', 'MOM.A.FLUX'] ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_h5_swe_00000000_v01.skt
            variables = ['datetime'] + [f'FLUX_{i}'for i in range(1,16)]+ [f'ENERGY_{i}' for i in range(1,16)] + ['Vpt', 'Jpx', 'Jpy', 'Jpz','Pp_XX', 'Pp_YY', 'Pp_ZZ', 'Pp_XY', 'Pp_XZ', 'Pp_YZ', 'Jax', 'Jay', 'Jaz']
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_plsp/{date[:4]}/wi_plsp_3dp_{date}_v02.cdf'#https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_sfpd_3dp_00000000_v01.skt
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

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
    def TDP_SOSP(self, scrap_date):
        try:
            csv_file = './data/WIND/TDP/SOSP/data.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = ['FLUX', 'ENERGY'] ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_h5_swe_00000000_v01.skt
            variables = ['datetime'] + [phy_obs[i] + f'_{k}' for i in range(2) for k in range(1,10)]
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_sosp/{date[:4]}/wi_sosp_3dp_{date}_v01.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []

                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

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
    def TDP_SOPD(self, scrap_date):
        try:
            csv_file = './data/WIND/TDP/SOPD/data.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = 'FLUX'## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_h5_swe_00000000_v01.skt
            pitch_angles = [
                15,
                35,
                57,
                80,
                102,
                123,
                145,
                165
            ]
            energy_bands = [
                "70keV",
                "130keV",
                "210keV",
                "330keV",
                "550keV",
                "1000keV",
                "2100keV",
                "4400keV",
                "6800keV"
            ]
            variables = ['datetime'] + [f'Proton_flux_{deg}_{ener}' for deg in pitch_angles for ener in energy_bands ]
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_sopd/{date[:4]}/wi_sopd_3dp_{date}_v02.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data = cdf_file[phy_obs][:]

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data  = np.concatenate([epoch, data.reshape((data.shape[0], data.shape[1]*data.shape[2]))], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def TDP_ELSP(self, scrap_date):
        try:
            csv_file = './data/WIND/TDP/ELSP/data.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = ['FLUX', 'ENERGY'] ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_h5_swe_00000000_v01.skt
            energy_bands = [
                '1113eV' ,
                '669.2eV',
                '426.8eV',
                '264.8eV',
                '165eV'  ,
                '103.3eV',
                '65.25eV',
                '41.8eV' ,
                '27.25eV',
                '18.3eV',
                '12.8eV',
                '9.4eV' ,
                '7.25eV',
                '5.9eV' ,
                '5.2eV' 
            ]
            variables = ['datetime'] + [f'electron_{phy_obs[i]}_{ener}' for i in range(2) for ener in energy_bands] #https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_elsp_3dp_00000000_v01.skt
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_elsp/{date[:4]}/wi_elsp_3dp_{date}_v01.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []
                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

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
    def TDP_ELPD(self, scrap_date):
        try:
            csv_file = './data/WIND/TDP/ELPD/data.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = 'FLUX' ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_elpd_3dp_00000000_v01.skt
            pitch_angles = [
                15,
                35,
                57,
                80,
                102,
                123,
                145,
                165
            ]
            energy_bands = [
                '1150eV',
                '790eV',
                '540eV',
                '370eV',
                '255eV',
                '175eV',
                '121eV',
                '84eV',
                '58eV',
                '41eV',
                '29eV',
                '20.5eV',
                '15eV',
                '11.3eV',
                '8.6eV'
            ]
            variables = ['datetime'] + [f'electron_flux_{deg}_{ener}' for deg in pitch_angles for ener in energy_bands] #https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_elsp_3dp_00000000_v01.skt
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_elpd/{date[:4]}/wi_elpd_3dp_{date}_v02.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data = cdf_file[phy_obs][:]

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data  = np.concatenate([epoch, data.reshape((data.shape[0], data.shape[1]*data.shape[2]))], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def TDP_EHSP(self, scrap_date):
        try:
            csv_file = './data/WIND/TDP/EHSP/data.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = ['FLUX', 'ENERGY'] ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_ehsp_3dp_00000000_v01.skt
            energy_bands = [
                '27660eV' ,
                '18940eV' ,
                '12970eV' ,
                '8875eV'  ,
                '6076eV'  ,
                '4161eV'  ,
                '2849eV'  ,
                '1952eV'  ,
                '1339eV'  ,
                '920.3eV',
                '634.4eV',
                '432.7eV',
                '292.0eV',
                '200.1eV',
                '136.8eV',
            ]
            variables = ['datetime'] + [f'electron_{phy_obs[i]}_{ener}' for i in range(2) for ener in energy_bands] 
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_ehsp/{date[:4]}/wi_ehsp_3dp_{date}_v02.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []
                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

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
    def TDP_EHPD(self, scrap_date):
        try:
            csv_file = './data/WIND/TDP/EHPD/data.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = 'FLUX' ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_ehpd_3dp_00000000_v01.skt
            pitch_angles = [
                15,
                35,
                57,
                80,
                102,
                123,
                145,
                165
            ]
            energy_bands = [
                '27660eV' ,
                '18940eV' ,
                '12970eV' ,
                '8875eV'  ,
                '6076eV'  ,
                '4161eV'  ,
                '2849eV'  ,
                '1952eV'  ,
                '1339eV'  ,
                '920.3eV',
                '634.4eV',
                '432.7eV',
                '292.0eV',
                '200.1eV',
                '136.8eV',
            ]
            variables = ['datetime'] + [f'electron_flux_{deg}_{ener}' for deg in pitch_angles for ener in energy_bands] #https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_ehpd_3dp_00000000_v01.skt
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_ehpd/{date[:4]}/wi_ehpd_3dp_{date}_v02.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data = cdf_file[phy_obs][:]

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data  = np.concatenate([epoch, data.reshape((data.shape[0], data.shape[1]*data.shape[2]))], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def TDP_SFSP(self, scrap_date):
        try:
            csv_file = './data/WIND/TDP/SFSP/data.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = ['FLUX', 'ENERGY'] ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_h5_swe_00000000_v01.skt
            energy_bands = [
                '27keV', 
                '40keV', 
                '86keV', 
                '110keV',
                '180keV',
                '310keV',
                '520keV'
            ]
            variables = ['datetime'] + [f'electron_{phy_obs[i]}_{ener}' for i in range(2) for ener in energy_bands] #https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_sfsp_3dp_00000000_v01.skt
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_sfsp/{date[:4]}/wi_sfsp_3dp_{date}_v01.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []
                for var in phy_obs:
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

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
    def TDP_SFPD(self, scrap_date):
        try:
            csv_file = './data/WIND/TDP/SFPD/data.csv' #directories
            temp_root = './data/WIND/TDP/temp' 
            os.makedirs(temp_root)
            os.makedirs(csv_file[:-9]) #create folder
            phy_obs = 'FLUX' ## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_elpd_3dp_00000000_v01.skt
            pitch_angles = [
                15,
                35,
                57,
                80,
                102,
                123,
                145,
                165
            ]
            energy_bands = [
                '27keV', 
                '40keV', 
                '86keV', 
                '110keV',
                '180keV',
                '310keV',
                '520keV'
            ]
            variables = ['datetime'] + [f'electron_flux_{deg}_{ener}' for deg in pitch_angles for ener in energy_bands] #https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_elsp_3dp_00000000_v01.skt
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/sp_phys/data/wind/3dp/3dp_sfpd/{date[:4]}/wi_sfpd_3dp_{date}_v02.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data = cdf_file[phy_obs][:]

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data  = np.concatenate([epoch, data.reshape((data.shape[0], data.shape[1]*data.shape[2]))], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
            shutil.rmtree(temp_root)
        except FileExistsError:
            pass
        df = pd.read_csv(csv_file, parse_dates=['datetime'], index_col='datetime')
        return df
    def SMS(self, scrap_date):
        try:
            csv_file = './data/WIND/SMS/data.csv' #directories
            temp_root = './data/WIND/SMS/temp' 
            os.makedirs(temp_root)
            angle = [
                53,
                0,
                -53
            ]
            phy_obs = ['counts_tc_he2plus', 'counts_tc_heplus', 'counts_tc_hplus', 'counts_tc_o6plus', 'counts_tc_oplus', 'counts_tc_c5plus', 'counts_tc_fe10plus', 
                        'dJ_tc_he2plus', 'dJ_tc_heplus', 'dJ_tc_hplus', 'dJ_tc_o6plus', 'dJ_tc_oplus', 'dJ_tc_c5plus', 'dJ_tc_fe10plus']## metadata: https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_l2-3min_sms-stics-vdf-solarwind_00000000_v01.skt
            variables = ['datetime'] + [phy_obs[i] + f'_{deg}'for i in range(len(phy_obs)) for deg in angle] #https://cdaweb.gsfc.nasa.gov/pub/software/cdawlib/0SKELTABLES/wi_l2-3min_sms-stics-vdf-solarwind_00000000_v01.skt
            with open(csv_file, 'w') as file:
                file.writelines(','.join(variables) + '\n')
            for date in scrap_date:
                url = f'https://cdaweb.gsfc.nasa.gov/data/wind/sms/l2/stics_cdf/3min_vdf_solarwind/{date[:4]}/wi_l2-3min_sms-stics-vdf-solarwind_{date}_v01.cdf'
                name = date + '.cdf'
                download_url(url, temp_root, name)
                cdf_path = os.path.join(temp_root, name)
                cdf_file = pycdf.CDF(cdf_path)

                data_columns = []
                for var in phy_obs:
                    data_columns.append(np.mean(np.mean(cdf_file[var][:], axis = 1), axis=2))

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

        celias_sem = pd.read_csv(csv_root, parse_dates=['datetime'], index_col='datetime', date_format='%y-%m-%d %H:%M:%S')
        celias_sem.index = pd.to_datetime(celias_sem.index)
        celias_sem = celias_sem.sort_index()
        celias_sem = celias_sem.loc[scrap_date[0]: scrap_date[-1]]
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

        df = pd.read_csv(csv_root, parse_dates=['datetime'], index_col='datetime', date_format='%y-%m-%d %H:%M:%S').sort_index()
        df = df.loc[scrap_date[0]:scrap_date[-1]]
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
        costep_ephin = pd.read_csv(csv_root, parse_dates=['datetime'], index_col='datetime', date_format='%Y-%m-%d %H:%M:%S').sort_index()
        costep_ephin = costep_ephin.loc[scrap_date[0]:scrap_date[-1]]
        return costep_ephin
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
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

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
            variables = ['datetime'] + ['Bnorm', 'BGSM_x', 'BGSM_y', 'BGSM_z', 'SC_GSM_x', 'SC_GSM_y', 'SC_GSM_z', 'dBrms', 'BGSE_x', 'BGSE_y', 'BGSE_z', 'SC_GSE_x', 'SC_GSE_y', 'SC_GSE_z']
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
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

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
            temp_root = './data/ACE/SWEPAM/temp' 
            os.makedirs(temp_root) #create folder
            phy_obs = ['Np', 'Vp','Tpr','alpha_ratio', 'V_GSE', 'V_GSM'] #variables#change
            variables = ['datetime'] + phy_obs[:4] + [phy_obs[i] + f'_{k}' for i in range(4,6) for k in range(1,4)]
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
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

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
            temp_root = './data/ACE/SWICS/temp' 
            os.makedirs(temp_root) #create folder
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
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

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
            phy_obs = [
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
                    cond = cdf_file[var][:].reshape(-1,1) if len(cdf_file[var][:].shape) == 1 else cdf_file[var][:]
                    data_columns.append(cond)

                epoch = np.array([str(date.strftime('%Y-%m-%d %H:%M:%S.%f')) for date in cdf_file['Epoch'][:]]).reshape(-1,1)
                data = np.concatenate(data_columns, axis = 1, dtype =  np.float32)
                data  = np.concatenate([epoch, data], axis = 1)
                with open(csv_file, 'a') as file:
                    np.savetxt(file, data, delimiter=',', fmt='%s')
            shutil.rmtree(temp_root)
        except:
            pass
        epam = pd.read_csv(csv_file, parse_dates=['datetime'], index_col = 'datetime')
        return epam
class DSCOVR:
    def MAGFC(self, scrap_date: list, sep = False):
        os.makedirs('data/compressed', exist_ok=True)
        os.makedirs('data/uncompressed', exist_ok=True)
        os.makedirs('data/DSCOVR_L2/faraday', exist_ok=True)
        os.makedirs('data/DSCOVR_L1/faraday', exist_ok=True)
        os.makedirs('data/DSCOVR_L2/magnetometer', exist_ok=True)
        os.makedirs('data/DSCOVR_L1/magnetometer', exist_ok=True)
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
            if filename[:-6]+'.csv' in os.listdir('data/DSCOVR_L2/faraday') or filename[:-6]+'.csv' in os.listdir('data/DSCOVR_L1/faraday') or filename[:-6]+'.csv' in os.listdir('data/DSCOVR_L1/magnetometer') or filename[:-6]+'.csv' in os.listdir('data/DSCOVR_L2/magnetometer'):
                continue
            elif filename[:-3] in os.listdir('data/uncompressed'):
                output_file = os.path.join('data/uncompressed',filename)[:-3]
                pass
            else:
                download_url(url, root, filename)
                file = os.path.join(root,filename)
                output_file = os.path.join('data/uncompressed',filename)[:-3]
                with gzip.open(file, 'rb') as compressed_file:
                        with open(output_file, 'wb') as decompressed_file:
                            decompressed_file.write(compressed_file.read())
                os.remove(file)
            if 'fc1' in filename:
                dataset = xr.open_dataset(output_file)

                df = dataset.to_dataframe()

                dataset.close()

                important_variables = ['proton_vx_gse', 'proton_vy_gse', 'proton_vz_gse', 'proton_speed', 'proton_density', 'proton_temperature']

                faraday_cup = df[important_variables]

                faraday_cup = faraday_cup.resample('1min').mean()

                faraday_cup.to_csv(f'data/DSCOVR_L1/faraday/{filename[:-6]}.csv')

                os.remove(output_file)

            elif 'f1m' in filename:
                dataset = xr.open_dataset(output_file)

                df = dataset.to_dataframe()

                dataset.close()

                important_variables = ['proton_vx_gse', 'proton_vy_gse', 'proton_vz_gse', 'proton_speed', 'proton_density', 'proton_temperature']

                faraday_cup = df[important_variables]
                ##Feature engineering

                faraday_cup.to_csv(f'data/DSCOVR_L2/faraday/{filename[:-6]}.csv')
                
                os.remove(output_file)
            elif 'm1m' in filename:
                dataset = xr.open_dataset(output_file)

                df = dataset.to_dataframe()

                dataset.close()

                important_variables = ['bx_gse', 'by_gse', 'bz_gse', 'theta_gse', 'phi_gse']

                magnetometer = df[important_variables]

                magnetometer.to_csv(f'data/DSCOVR_L2/magnetometer/{filename[:-6]}.csv')
                
                os.remove(output_file)
            else:
                dataset = xr.open_dataset(output_file)

                df = dataset.to_dataframe()

                dataset.close()

                important_variables = ['bx_gse', 'by_gse', 'bz_gse', 'theta_gse', 'phi_gse']

                magnetometer = df[important_variables]

                magnetometer = magnetometer.resample('1min').mean()

                magnetometer.to_csv(f'data/DSCOVR_L1/magnetometer/{filename[:-6]}.csv')
                os.remove(output_file)

        start_time =scrap_date[0]
        end_time = scrap_date[-1]

        level_1, level_2 = self.from_csv(start_time, end_time, sep)

        return level_1, level_2

    def from_csv(self, start_time, end_time, sep = False):
        fc1_list = []
        for file in os.listdir('data/DSCOVR_L1/faraday'):
            file = os.path.join('data/DSCOVR_L1/faraday', file)
            data = pd.read_csv(file, index_col=0)
            fc1_list.append(data)
        
        mg1_list = []
        for file in os.listdir('data/DSCOVR_L1/magnetometer'):
            file = os.path.join('data/DSCOVR_L1/magnetometer', file)
            data = pd.read_csv(file, index_col=0)
            mg1_list.append(data)
        
        f1m_list = []
        for file in os.listdir('data/DSCOVR_L2/faraday'):
            file = os.path.join('data/DSCOVR_L2/faraday', file)
            data = pd.read_csv(file, index_col=0)
            f1m_list.append(data)
        
        m1m_list = []
        for file in os.listdir('data/DSCOVR_L2/magnetometer'):
            file = os.path.join('data/DSCOVR_L2/magnetometer', file)
            data = pd.read_csv(file, index_col=0)
            m1m_list.append(data)

        fc1 = pd.concat(fc1_list)
        mg1 = pd.concat(mg1_list)
        f1m = pd.concat(f1m_list)
        m1m = pd.concat(m1m_list)
        fc1 = fc1[~fc1.index.duplicated(keep='first')]
        mg1 = mg1[~mg1.index.duplicated(keep='first')]
        f1m = f1m[~f1m.index.duplicated(keep='first')]
        m1m = m1m[~m1m.index.duplicated(keep='first')]
        fc1.index = pd.to_datetime(fc1.index)
        mg1.index = pd.to_datetime(mg1.index)
        f1m.index = pd.to_datetime(f1m.index)
        m1m.index = pd.to_datetime(m1m.index)
        start_time_ = f'{start_time[:4]}-{start_time[4:6]}-{start_time[-2:]} 00:00:00'
        end_time_ = f'{end_time[:4]}-{end_time[4:6]}-{end_time[-2:]} 23:59:00' 
        freq = '1T'
        full_time_index = pd.date_range(start=start_time_, end=end_time_, freq=freq)
        fc1 = fc1.reindex(full_time_index).interpolate(method = 'linear')
        mg1 = mg1.reindex(full_time_index).interpolate(method = 'linear')
        f1m = f1m.reindex(full_time_index).interpolate(method = 'linear')
        m1m = m1m.reindex(full_time_index).interpolate(method = 'linear')
        if sep:
            level_1 = (fc1, mg1)
            level_2 = (f1m, m1m)
        else:
            level_1 = pd.concat([fc1, mg1], axis =1)
            level_2 = pd.concat([f1m, m1m], axis =1)
            
            level_1.index = pd.to_datetime(level_1.index)
            level_2.index = pd.to_datetime(level_2.index)
        
        return level_1, level_2

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