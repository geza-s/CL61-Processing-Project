# system management
import glob
import os
import shutil
import tempfile
import importlib

# Array
import xarray as xr
import numpy as np
import pandas as pd

# Visualize
from matplotlib import pyplot as plt
import cmcrameri.cm as cmc  # batlow colourmap

# Signal processing (check if necessary)
from scipy import signal
from scipy.interpolate import BSpline

# Other (to check if necessary)
import dask
from tqdm import tqdm
import seaborn as sns

# Functions implemented for data processing and visualization
from CL61_module import visualization
from CL61_module import process

from .utils import load_config

plt.style.use('bmh')
COLOR_MAP_NAME = 'cmc.batlow'
COLOR_MAP = cmc.batlow  # type: ignore

class CL61Processor:
    def __enter__(self):
        # Getting the temporary folder
        self.temp_folder = tempfile.mkdtemp()

    def __init__(self, folder_path,
                 start_datetime=None, end_datetime=None,
                 specific_filename=None,
                 config_path='config_classification.yml',
                 transfer_files_locally = False,
                 parallel_computing = True,
                 dataset = None):
        
        # Folder paths
        self.folder_path = folder_path
        self.module_dir = os.path.dirname(__file__)
        # Congig elements 
        self.config_filepath = config_path
        self.classification_config = self.load_config(config_path)
        # Dataset and time elements
        self.dataset = dataset
        self.file_time_mapping = None
        self.period_start = start_datetime
        self.period_end = end_datetime
        # Temporary folder
        self.temp_folder = None

        # Initialize dataset based on conditions set
        if self.dataset is None:
            self._load_data(specific_filename, transfer_files_locally, parallel_computing)

        # Manage metadata
        self._set_metadata()

        # Takes the major subclasses
        self.plot = visualization.PlotCL61(self.dataset)
        self.process = process.ProcessCL61(self)


    def _load_data(self, specific_filename, transfer_files_locally, parallel_computing):
        if specific_filename:
            self.load_specific_data(specific_filename)
        else:
            filepaths_interest_df = self.get_filepaths_with_timestamps(period_start=self.period_start,
                                                                       period_end=self.period_end)
            if transfer_files_locally:
                self._transfer_files_locally(filepaths_interest_df)
                # update folder and filepaths
                self.folder_path = self.temp_folder  
                self.get_filepaths_with_timestamps() 

            self.dataset = self.load_netcdf4_data_in_timeperiod(parallel_computing=parallel_computing)

    def _set_metadata(self):
        self.dataset['beta_att'].attrs['name'] = 'attenuated backscatter coefficient'
        self.dataset['linear_depol_ratio'].attrs['name'] = 'linear depolarisation ratio'


    def _transfer_files_locally(self, file_names_to_copy,
                                 temp_folder = None):
        """
        Transfers all filenames mentioned from the data source folder to a temporary folder

        Args:
            file_names_to_copy (list): list of filenames in source folder to transfer
        """
        # Define source and end folders
        data_folder = self.folder_path
        temp_folder = temp_folder or self.temp_folder

        # Copy each file to the local folder
        print(f"File transfer to temporary folder {temp_folder}")
        for file_name in tqdm(file_names_to_copy):
            source_path = os.path.join(data_folder, file_name)
            destination_path = os.path.join(temp_folder, file_name)
            
            try:
                shutil.copy2(source_path, destination_path)
                print(f"File '{file_name}' copied successfully.")
            except Exception as e:
                print(f"Error copying '{file_name}': {e}")

        return temp_folder

    def load_specific_data(self, specific_filename):
        '''
        Loads specific netcdf data as whole dataset
        args:
            specific_filename: filename of the 
        '''
        specific_filepath = os.path.join(self.folder_path, specific_filename)
        self.dataset = xr.open_dataset(specific_filepath)
        self.period_start = self.dataset['time'][0].values
        self.period_end = self.dataset['time'][-1].values
        self.file_time_mapping = None


    def load_config(self, config_path):
        '''
        Loads a yml config file from given relative or absolute config file path
        Args:
            config_path: relative or absolute config file path
        '''
        # Check if the given config_path is an absolute path
        if os.path.isabs(config_path):
            if os.path.exists(config_path):
                return load_config(config_path)
            else:
                raise ValueError("Config file does not exist: " + config_path)
        else:
            # Join config_path with module_dir
            config_full_path = os.path.join(self.module_dir, config_path)
            if os.path.exists(config_full_path):
                self.config_filepath = config_full_path  # just update the filepath to absolute path
                return load_config(config_full_path)
            else:
                raise ValueError("Config file does not exist in module_dir: " + config_full_path)

    def get_filepaths_with_timestamps(self, period_start = None, period_end=None):
        """
        Gets all netCDF file paths in the given folder.
        Saves the results into a DataFrame in variable self.file_time_mapping.

        Args:
            period_start (datetime string, optional): Start of period of interest. Defaults to None.
            period_end (datetime string, optional): End of period of interest. Defaults to None.

        Returns:
            pandas dataframe : a slice of the filepaths mapping dataframe for the given slice of time
        """
        filenames = glob.glob('*.nc', root_dir=self.folder_path)
        filepaths = glob.glob(os.path.join(self.folder_path, '*.nc'))
        timestamps = [pd.to_datetime(path[-18:-3], format='%Y%m%d_%H%M%S') for path in filepaths]

        self.file_time_mapping = pd.DataFrame({'filenames': filenames, 'file_name_path': filepaths}, index=timestamps)
        self.file_time_mapping.sort_index(inplace=True)

        period_start = period_start or self.file_time_mapping.index[0]
        period_end = period_end or self.file_time_mapping.index[-1]

        return self.file_time_mapping.loc[period_start:period_end]

    def load_netcdf4_data_in_timeperiod(self, parallel_computing = False, chunksize = 100):
        """
        Load the netcdf4 files at given folder location and inside the given period.

        Args:
            parrallel_computing (bool, optional): If dask is installed, open_mfdataset can increase in some case performance. Defaults to False.

        Returns:
            _type_: _description_
        """
        # Select all files in folder referering to wished period
        selected_filemaps = self.file_time_mapping.loc[self.period_start:self.period_end]
        
        if parallel_computing:
            print(selected_filemaps)
            return xr.open_mfdataset(selected_filemaps['file_name_path'],
                                     engine = "netcdf4",
                                     parallel = True)
        else:
            combined_dataset = None
            for row in tqdm(selected_filemaps.iterrows(), total=selected_filemaps.shape[0]):
                # Open file alone to xarray
                row_array = xr.open_dataset(row[1]['file_name_path'], chunks='auto')
                # Combine the datasets
                if combined_dataset is None:
                    combined_dataset = row_array
                else:
                    combined_dataset = xr.concat([combined_dataset, row_array], dim='time')
            return combined_dataset

    def get_subset(self, start_time, end_time):
        """
        Get a subset of the dataset for the specified time range as a new CL61Processor instance.

        Args:
            start_time (str): Start time (e.g., 'YYYY-MM-DD HH:MM:SS').
            end_time (str): End time (e.g., 'YYYY-MM-DD HH:MM:SS').

        Returns:
            A new CL61Processor instance with the subset of data.
        """
        if self.dataset is None:
            raise ValueError("No data loaded yet. Call load_data_in_timeperiod or load_specific_data first.")

        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        subset = self.dataset.sel(time=slice(start_time, end_time)).copy()

        # Create a new CL61Processor instance with the subset data
        subset_processor = CL61Processor(self.folder_path,
                                         start_datetime=start_time,
                                         end_datetime=end_time,
                                         config_path=self.config_filepath,
                                         dataset=subset)

        return subset_processor

    def reload_modules(self):
        importlib.reload(visualization)
        importlib.reload(process)
        return

    def __exit__(self):
        # Close dataset and remove temporary folder
        if isinstance(self.dataset, xr.Dataset):
            self.dataset.close()
        shutil.rmtree(self.temp_folder)