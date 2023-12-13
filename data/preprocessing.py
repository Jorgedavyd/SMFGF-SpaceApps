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
import shutil

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