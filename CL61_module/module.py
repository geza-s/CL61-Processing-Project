# system management
import os
import argparse
import glob
import shutil

# Array
import xarray as xr
import numpy as np
import pandas as pd

# Manage time
from datetime import datetime

# Visualize
from matplotlib import pyplot as plt
import cmcrameri.cm as cmc  # batlow colourmap

# Signal processing (check if necessary)
#from scipy import signal
#from scipy.interpolate import BSpline

# Other (to check if necessary)
#import dask
from tqdm import tqdm
#import seaborn as sns

# Functions implemented for data processing and visualization
from CL61_module import visualization
from CL61_module import process
from CL61_module import classification

from .utils import load_config

plt.style.use('bmh')
COLOR_MAP_NAME = 'cmc.batlow'
COLOR_MAP = cmc.batlow  # type: ignore

class CL61Processor:
    def __enter__(self):
        return
    
    def __init__(self, folder_path,
                 start_datetime=None, end_datetime=None,
                 specific_filename=None,
                 config_path='config_classification.json',
                 load_to_memory = False,
                 transfer_files_locally = False,
                 parallel_computing = False,
                 dataset = None,
                 verbose = 1):
        """
        Initialize CL61Processor instance.

        Parameters:
        - folder_path: str, required, path to the data folder.
        - start_datetime: str, optional, start date and time in the format YYYY-MM-DD HH:mm:ss.
        - end_datetime: str, optional, end date and time in the format YYYY-MM-DD HH:mm:ss.
        - specific_filename: str, optional, specific filename to process.
        - config_path: str, optional, path to the configuration file (default='config_classification.json').
        - load_to_memory: bool, optional, load data into memory (default=False).
        - transfer_files_locally: bool, optional, transfer files locally (default=False).
        - parallel_computing: bool, optional, use parallel computing (default=False).
        - dataset: object, optional, pre-loaded dataset.
        - verbose: int, optional, verbosity level for debugging (default=1).

        Raises:
        - ValueError: If the 'verbose' parameter is not an integer.
        """
        # verbose parameter for debugging
        if not isinstance(verbose, int):
                raise ValueError("The 'verbose' parameter must be an integer.")
        else:
            self.verbose = verbose
        
        # Folder paths
        self.folder_path = folder_path
        self.module_dir = os.path.dirname(__file__)
        
        # Getting the temporary folder
        self.temp_dir = os.path.join(self.module_dir, "temp")
        os.makedirs(self.temp_dir, exist_ok=True)
        if transfer_files_locally:
            self.debug_print(f"Created temporary folder at: {self.temp_dir}")

        # Config filepath
        self.config_path = config_path

        # Dataset and time elements
        self.dataset = dataset
        self.period_start = start_datetime
        self.period_end = end_datetime

        # Get Info on files in folder
        self.files_in_data_folder = None
        self.files_in_time_periods = None
        if specific_filename is None:
            self.debug_print("Getting filenames in data folder ... ")
            self.files_in_time_periods = self.get_filepaths_with_timestamps(start_datetime, end_datetime)
            self.debug_print(self.files_in_time_periods['filenames'], level=3)
            
        # Initialize dataset based on conditions set
        if self.dataset is None:
            self.debug_print((f"Loading dataset from files with parameter: \n \t load_to_memory={load_to_memory},"
                             f"transfer_files_locally={transfer_files_locally}, parallel_computing = {parallel_computing}"))
            self._load_data(specific_filename,
                            load_to_memory = load_to_memory,
                            transfer_files_locally=transfer_files_locally,
                            parallel_computing=parallel_computing)

        # Manage metadata
        self._set_metadata()

        # Takes the major subclasses
        self.plot = visualization.PlotCL61(dataset = self.dataset, parent = self)
        self.process_noise = process.NoiseProcessor(parent = self)
        self.classification = classification.CL61Classifier(parent = self)

        self.debug_print(message = "Module initilaization completed !", level=1)
        return 

    def _load_data(self, specific_filename, load_to_memory, transfer_files_locally, parallel_computing):
        '''
        Manage different data loading situation:
        a) from specific file, b) from folder with original files, c) with temporary transfer to local folder
        '''
        if specific_filename:
            self.debug_print(f"load data from specific filne {specific_filename}")
            self.load_specific_data(specific_filename)
        else:
            if transfer_files_locally:
                self._transfer_files_locally(self.files_in_time_periods['filenames'])
                # update folder and filepaths
                self.folder_path = self.temp_dir  
                self.files_in_time_periods = self.get_filepaths_with_timestamps(period_start=self.period_start,
                                                                                 period_end=self.period_end)
            # Opening all files
            self.dataset = self.load_netcdf4_data_in_timeperiod(parallel_computing=parallel_computing,
                                                                load_to_memory=load_to_memory)
        return

    def _transfer_files_locally(self, file_names_to_copy, temp_dir=None):
        """
        Transfers all filenames mentioned from the data source folder to a temporary folder

        Args:
            file_names_to_copy (list): list of filenames in source folder to transfer
        """
        # Define source and end folders
        data_folder = self.folder_path
        temp_dir = temp_dir or self.temp_dir

        self.debug_print(f"File transfer from {data_folder} to temporary folder {temp_dir}", level=1)
        # Copy each file to the local folder
        for file_name in tqdm(file_names_to_copy):
            source_path = os.path.join(data_folder, file_name)
            dest_path = os.path.join(temp_dir, file_name)
            self.debug_print(f'Copying file from {source_path} to {dest_path}', level=3)
            try:
                shutil.copy(src = source_path, dst = dest_path)
            except shutil.SameFileError:
                print(f"File '{file_name}' already exists at destination. Skipping copy.")
                continue
            except FileNotFoundError:
                print(f"Error copying '{file_name}': Source file not found: {source_path}")
                
        return temp_dir

    def _set_metadata(self):
        self.dataset['beta_att'].attrs['name'] = 'attenuated backscatter coefficient'
        self.dataset['linear_depol_ratio'].attrs['name'] = 'linear depolarisation ratio'

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
        self.files_in_data_folder = None
        return
    

    def get_filepaths_with_timestamps(self, period_start = None, period_end=None):
        """
        Gets all netCDF file paths in the given folder.
        Saves the results into a DataFrame in variable self.files_in_data_folder.

        Args:
            period_start (datetime string, optional): Start of period of interest. Defaults to None.
            period_end (datetime string, optional): End of period of interest. Defaults to None.

        Returns:
            pandas dataframe : a slice of the filepaths mapping dataframe for the given slice of time
        """
        self.debug_print('Getting all filenames from folder with glob *.nc', level=2)
        filenames = glob.glob('*.nc', root_dir=self.folder_path)
        filepaths = glob.glob(os.path.join(self.folder_path, '*.nc'))

        timestamps = [pd.to_datetime(path[-18:-3], format='%Y%m%d_%H%M%S') for path in filepaths]

        self.debug_print('Creating associated dataframe', level=2)
        self.files_in_data_folder = pd.DataFrame({'filenames': filenames, 'file_name_path': filepaths}, index=timestamps)
        self.files_in_data_folder.sort_index(inplace=True)

        initial_time_index = self.files_in_data_folder.index[0]
        end_time_index = self.files_in_data_folder.index[-1]

        period_start = period_start or initial_time_index
        period_end = period_end or end_time_index

        # Convert start and end dates to Timestamp objects
        if isinstance(period_start, str):
            period_start = pd.to_datetime(period_start)
        if isinstance(period_end, str):
            period_end = pd.to_datetime(period_end)

        if period_start < initial_time_index:
            if period_end < initial_time_index:
                raise(ValueError(f"Period start and period end are found outside of range of files: {initial_time_index} to {end_time_index}"))
            else:
                period_start = initial_time_index

        if period_end > end_time_index:
            if period_start> end_time_index:
                raise(ValueError(f"Period start and period end are found outside of range of file: {initial_time_index} to {end_time_index}"))
            else:
                period_end = end_time_index


        # Select all rows between the two dates
        selected_rows = self.files_in_data_folder.loc[(self.files_in_data_folder.index >= period_start) 
                                                   & (self.files_in_data_folder.index <= period_end)]

        return selected_rows


    def load_netcdf4_data_in_timeperiod(self, parallel_computing = False, load_to_memory = False):
        """
        Load the netcdf4 files at given folder location and inside the given period.

        Args:
            parrallel_computing (bool, optional): If dask is installed, open_mfdataset can increase in some case performance. Defaults to False.

        Returns:
            _type_: _description_
        """
        # Select all files in folder referering to wished period
        selected_files = self.files_in_time_periods['file_name_path']
        if len(selected_files)==0:
            raise ValueError("Error, no related files found in given folder for given time range")
        
        if parallel_computing:
            self.debug_print(f"Opening all in one with xr open_mfdataset from file {selected_files[0]}", level=2)
            dataset = xr.open_mfdataset(selected_files,
                                     chunks={'time': 'auto'},
                                     engine = "netcdf4",
                                     parallel = True)
        else:
            self.debug_print(f"Opening {len(selected_files)} files from given folder", level=2)
            dataset = None # variable to store dataset
            for filepath_i in tqdm(selected_files, total=len(selected_files)):
                # Open file alone to xarray
                row_array = xr.open_dataset(filepath_i, chunks='auto')
                # Combine the datasets
                if dataset is None:
                    dataset = row_array
                else:
                    dataset = xr.concat([dataset, row_array], dim='time')
            
        
        if load_to_memory:
            self.debug_print('Loading dataset into memory', level=2)
            return dataset.load()
        else :
            return dataset
        

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

    def auto_process(self, plot_results = True, clustering_method = 'kmean', range_limits = [0,10000]):
        ''' Runs automated pipeline based on default values'''

        # Show timeserie before noise removal
        if plot_results:
            self.plot.show_timeserie(range_limits=range_limits)

        # Remove noise and show results afterwards
        print("Step 1: applying noise removal process")
        self.process_noise.mask_noise()
        if plot_results:
            self.plot.show_timeserie(variable_names=['beta_att_clean', 'linear_depol_ratio_clean'], range_limits=range_limits)
        
        # Plot cloud base heights
        self.plot.show_cloud_base_heights(underlying_variable='beta_att_clean',
                                          range_limits=range_limits,
                                          save_fig=plot_results)
        
        # Cluster data then classify and show results
        print(f"step 2: cluster data with {clustering_method} method")
        if clustering_method == 'kmean':
            self.classification.Kmeans_clustering(cluster_N=16, plot_result=plot_results, plot_range=range_limits)
            self.classification.classify_clusters(cluster_variable='kmean_clusters')
        elif clustering_method == 'dbscan':
            print(f'This method is experimental and usually shows poor results...')
            self.classification.dbscan_clustering(dbscan_eps=0.3, plot_result=plot_results, plot_range=range_limits)
            self.classification.classify_clusters(cluster_variable='dbscan_clusters')
        else:
            raise(ValueError(f'clustering_method should be of {"kmean" or "dbscan"}; given value {clustering_method} not supported'))
        
        if plot_results:
            self.plot.show_classified_timeserie(classified_variable='classified_clusters', ylims = range_limits)

        # classify dirctly element-wise and show results
        print('Step 3: Classify the elements and clusters based on the classification given in config file')
        self.classification.classify_elementwise(beta_attenuation_varname='beta_att_clean',
                                                    linear_depol_ratio_varname='linear_depol_ratio_clean')
        if plot_results:
            self.plot.show_classified_timeserie(classified_variable='classified_elements', ylims = range_limits)
        return
        
    def debug_print(self, message, level=1):
        """Print debug messages based on the verbose level."""
        if self.verbose >= level:
            print(message)
        return
    
    def close(self):
        '''Apply exit with closing xarray and removing temporary folder'''
        return self.__exit__()

    def __exit__(self):
        # Close dataset and remove temporary folder
        if isinstance(self.dataset, xr.Dataset):
            self.dataset.close()
        #shutil.rmtree(self.temp_dir)


# Other Functions to run script directly from this file : -------------------------------------------------------------------

def parse_arguments():
    '''
    To be able to modify entry directly from command prompt
    '''
    parser = argparse.ArgumentParser(description='Process data using CL61Processor.')
    parser.add_argument('--data-folder', type=str,
                        default=r"X:\common\03_Experimental\Instruments\EERL_instruments\5. Remote sensing\CL61_Ceilometer\Data",
                        help='Path to the data folder')
    parser.add_argument('--start-time', type=str, default='2023-02-22 00:00:00',
                        help='Start date and time in the format YYYY-MM-DD HH:mm:ss')
    parser.add_argument('--end-time', type=str, default='2023-02-23 00:00:00',
                        help='End date and time in the format YYYY-MM-DD HH:mm:ss')
    parser.add_argument('--range', type=list, default = [0,15000],
                        help='List of measurement range (height) limits [min, max] for plots; can go from 0 to 15000 m.')
    return parser.parse_args()

def validate_data_folder(data_folder):
    while not os.path.exists(data_folder):
        print(f"Data folder '{data_folder}' does not exist.")
        data_folder = input("Enter a valid path to the data folder: ")
    return data_folder

def validate_datetime(datetime_str, datetime_name):
    try:
        # Perform additional validation if needed
        # For now, we're just checking the format
        datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        print(f"Invalid {datetime_name} format. Please use YYYY-MM-DD HH:mm:ss.")
        datetime_str = input(f"Enter a valid {datetime_name}: ")
        return validate_datetime(datetime_str, datetime_name)
    return datetime_str

if __name__ == "__main__":
    args = parse_arguments()

    # Validate data folder
    args.data_folder = validate_data_folder(args.data_folder)

    # Validate start time
    args.start_time = validate_datetime(args.start_time, 'start time')

    # Validate end time
    args.end_time = validate_datetime(args.end_time, 'end time')

    # Initialize CL61Processor
    cl61_processor = CL61Processor(folder_path=args.data_folder,
                                   start_datetime=args.start_time,
                                   end_datetime=args.end_time)

    # Perform data processing
    cl61_processor.auto_process(range_limits=args.range)
