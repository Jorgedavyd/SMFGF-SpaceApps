import xarray as xr
import pandas as pd
import os
import gzip

def gzip_to_nc():
    #defining raw data dataframes
    fc1 = []
    mg1 = []

    ##uncompress and save
    for file in os.listdir('compressed'):
        output_file = os.path.join('uncompressed/',file[:-3]) 
        file = os.path.join('compressed/',file)
        if 'oe_fc1' in file:
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
    
    return pd.concat(data_list).sort_index(inplace=True)

def concat_magnet(dataframes):
    
    data_list = []

    for nc_file in dataframes:
        dataset = xr.open_dataset(nc_file)

        df = dataset.to_dataframe()

        important_variables = ['bt','bx_gse', 'by_gse', 'bz_gse', 'theta_gse', 'phi_gse', 'bx_gsm','by_gsm', 'bz_gsm', 'theta_gsm', 'phi_gsm']

        magnetometer = df[important_variables]

        magnetometer = magnetometer.resample('1min').mean()

        data_list.append(magnetometer)
    
    return pd.concat(data_list).sort_index(inplace=True)

def main_preprocessing():
    fc1, mg1 = gzip_to_nc()
    faraday = faraday_preprocess(fc1)
    magnetometer = concat_magnet(mg1)
    return pd.concat([faraday, magnetometer], axis =1 )



