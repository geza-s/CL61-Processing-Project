# system management
import glob
import os
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
    def __init__(self, folder_path,
                 start_datetime=None, end_datetime=None,
                 specific_filename=None,
                 config_path='config_classification.yml',
                 dataset = None):
        
        # Path to the data folder
        self.folder_path = folder_path
        # Directory to the module folder 
        self.module_dir = os.path.dirname(__file__)
        # Open config file
        self.config_filepath = config_path
        self.classification_config = self.load_config(config_path)

        self.dataset = dataset
        self.file_time_mapping = None
        self.period_start = start_datetime
        self.period_end = end_datetime
        
        if self.dataset is None:
            if specific_filename:
                self.load_specific_data(specific_filename)
            else:
                self.file_time_mapping = self.get_filepaths_with_timestamps()
                self.dataset = self.load_netcdf4_data_in_timeperiod()

        # TAKES THE MAJOR SUBCLASSES
        self.plot = visualization.PlotCL61(self.dataset)
        self.process = process.ProcessCL61(self)

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

    def get_filepaths_with_timestamps(self):
        """
        Gets all netcdfs filepaths in the given folder

        Returns:
            A pandas dataframe with all the file_name_path indexed by datetime
        """
        # older version:
        # filepaths = glob.glob(self.folder_path + '/*.nc')
        # get_file_datetime = list(map(lambda date_str: pd.to_datetime(date_str, format='%Y%m%d_%H%M%S'),[path_name[-18:-3] for path_name in filepaths]))
        # df_data_files = pd.DataFrame([filepaths, get_file_datetime], index=['file_name_path', 'Datetime']).T
        # df_data_files.index = get_file_datetime
        # df_data_files = df_data_files.sort_index()

        filepaths = glob.glob(os.path.join(self.folder_path, '*.nc'))
        timestamps = [pd.to_datetime(path[-18:-3], format='%Y%m%d_%H%M%S') for path in filepaths]

        df_data_files = pd.DataFrame({'file_name_path': filepaths}, index=timestamps)
        df_data_files.sort_index(inplace=True)

        return df_data_files

    def load_netcdf4_data_in_timeperiod(self, parrallel_computing = False):
        """
        Load the netcdf4 files at given folder location and inside the given period.

        Args:
            parrallel_computing (bool, optional): If dask is installed, open_mfdataset can increase in some case performance. Defaults to False.

        Returns:
            _type_: _description_
        """
        # Select all files in folder referering to wished period
        selected_filemaps = self.file_time_mapping.loc[self.period_start:self.period_end]
        
        if parrallel_computing:
            return xr.open_mfdataset(selected_filemaps['file_name_path'], chunks={'time': 300})
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
        importlib.reload(noise_processing)
        return

    def close(self):
        """
        Close the NetCDF dataset.

        Returns:
            None
        """
        if self.dataset is not None:
            self.dataset.close()


class Process:
    def __init__(self, parent: CL61Processor):
        self.CL61_parent = parent

    def mask_noise(self, window_size=[7, 7]):
        """
        Process noise in the data by searching for negative values for beta_attenuation
        and creates new variables with suffix _clean in the dataset and a 'noise_mask'

        Args:
            window_size (int or list of int): size of the window for noise removal padding around null value.
        Returns:
            None
        """
        dataset = self.CL61_parent.dataset
        # Compute non-null masks
        beta_att_non_null_mask = noise_processing.non_noise_windows_mask(dataset,
                                                                         variable_name='beta_att',
                                                                         window_size=window_size,
                                                                         analysis_type='non-negative')

        linear_depol_non_null_mask = noise_processing.non_noise_windows_mask(dataset,
                                                                             variable_name='linear_depol_ratio',
                                                                             window_size=[1, 3],
                                                                             analysis_type='range')

        final_mask = beta_att_non_null_mask & linear_depol_non_null_mask
        to_interpolate_mask = beta_att_non_null_mask & ~linear_depol_non_null_mask

        print(
            'The results are stored under new variable: beta_att_clean, linear_depol_ratio_clean, noise mask and to_interpolate_mask')
        self.CL61_parent.dataset['beta_att_clean'] = xr.where(final_mask, dataset['beta_att'], np.nan)
        self.CL61_parent.dataset['linear_depol_ratio_clean'] = xr.where(final_mask, dataset['linear_depol_ratio'],
                                                                        np.nan)
        self.CL61_parent.dataset['noise_mask'] = xr.DataArray(data=final_mask, dims=['time', 'range'])
        self.CL61_parent.dataset['to_interpolate_mask'] = xr.DataArray(data=to_interpolate_mask, dims=['time', 'range'])

        return

    def perform_kmeans_clustering(self, variable_as_features=['beta_att', 'linear_depol_ratio'], weights=None,
                                  cluster_number=8, kmean_method='k-means++',
                                  plot_result=True, save_fig=True):
        """
        Perform k-means clustering on the dataset.

        Parameters:
        - variable_as_features: List of variable names to use as features (default: ['beta_att', 'linear_depol_ratio']).
        - weights: List of weights for feature columns (default: None).
        - cluster_number: The number of clusters (default: 8).
        - plot_result: If True, visualize the results (default: True).
        - kmean_method: The K-means initialization method (default: 'k-means++').
        - plot_result (Boolean): If results should be directly plotted or not
        - save_fig (Boolean or str): If True figure plotted should be saved. If str -> figure saved with name as given.

        Returns:
        - None
        """

        # Implement classification logic
        classification_result_array = classification.K_means_classifier(
            dataset=self.CL61_parent.dataset,
            variable_as_features=variable_as_features,
            weights=weights,
            cluster_N=cluster_number,
            plot_result=plot_result,
            kmean_method=kmean_method,
            save_fig=save_fig
        )

        new_var = 'kmean_clusters'
        print(f'saving results under following{new_var}')
        self.CL61_parent.dataset[new_var] = xr.DataArray(data=classification_result_array.T, dims=['time', 'range'])
        return

    def classify_clusters(self,
                          cluster_variable='kmean_clusters'):
        '''
        Classifies each cluster of cluster_variable array based on determined thresholds
        '''
        class_results, new_class_xarray = classification.threshold_classify_clusters(dataset=self.CL61_parent.dataset,
                                                                                     cluster_variable=cluster_variable)
        classified_cluster_var_name = 'classified_clusters'
        self.CL61_parent.dataset[classified_cluster_var_name] = new_class_xarray
        self.CL61_parent.cluster_class_map = class_results
        print(f" Successful cluster classification stored in dataset under {classified_cluster_var_name}")
        return

    def classify_elementwise(self,
                             variable_names=['beta_att_clean', 'linear_depol_ratio_clean']):
        '''
        Classifies directly element wise based on thresholds focused on the variables in question
        '''
        new_var_name = 'classified_elements'
        self.CL61_parent.dataset[new_var_name] = classification.classify_dataset(self.CL61_parent.dataset,
                                                                                 variable_names=variable_names)
        print(f" Successful pixel-wise classification stored in dataset under {new_var_name}")
        return
