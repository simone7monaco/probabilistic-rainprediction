import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import pandas as pd
import os

run_where="local"

if run_where=="local":
    root_path="/home/monaco/MultimodelPreci"
    scratch_path=root_path
elif run_where=="legion":
    root_path="/home/meteo/Monaco/MultimodelPreci"
    scratch_path="/mnt/netapp/scratch/LucaMonaco/MultimodelPreci"
elif run_where=="mafalda":
    root_path="/work/users/lmonaco/MultimodelPreci"
    scratch_path=root_path

input_path=scratch_path+"/case_study/24h_10mmMAX_radar"
tiff_path=input_path+"/obs/tiff"
obs_path=input_path+"/obs/data"

def get_files_with_full_path(directory):
    file_list = []
    year=[]
    month=[]
    day=[]
    for root, _, files in os.walk(directory):
        for file_name in files:
            temp=file_name.split(".")[0].split("-")
            file_path = os.path.join(root, file_name)
            file_list.append(file_path)
            year.append(temp[0])
            month.append(temp[1])
            day.append(temp[2])
    return zip(file_list,year,month,day)

# Call the function to obtain the list of files with full paths
files = get_files_with_full_path(tiff_path)
# Open the GeoTIFF file
for f in files:
    print(f[0])
    with rasterio.open(f[0]) as dataset:
        # Read the raster data
        raster = dataset.read(1)

        # Get the dimensions of the raster
        rows, cols = raster.shape

        metadata = dataset.meta
        crs = dataset.crs
        target_crs = 'EPSG:4326'

        transform, width, height = calculate_default_transform(crs, target_crs, dataset.width, dataset.height, *dataset.bounds)

        # Update the metadata with new CRS and transformation parameters
        metadata.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })    

        #transform = dataset.transform

        # Get the image dimensions
        height = dataset.height
        width = dataset.width

        # Generate coordinate matrices
        lon_matrix, lat_matrix = [], []
        for y in range(height):
            lon_row, lat_row = [], []
            for x in range(width):
                lon, lat = transform * (x, y)
                lon_row.append(lon)
                lat_row.append(lat)
            lon_matrix.append(lon_row)
            lat_matrix.append(lat_row)

        lat_matrix=np.array(lat_matrix)
        lon_matrix=np.array(lon_matrix)
        raster=np.array(raster)
               
        pd.DataFrame(raster).to_csv(obs_path+"/radarPIEM_"+f[1]+f[2]+f[3]+"_raw.csv", sep=';', index=False, header=False)
        pd.DataFrame(lat_matrix).to_csv(obs_path+"/radarPIEM_lat_raw.csv", sep=';', index=False, header=False)
        pd.DataFrame(lon_matrix).to_csv(obs_path+"/radarPIEM_lon_raw.csv", sep=';', index=False, header=False)
