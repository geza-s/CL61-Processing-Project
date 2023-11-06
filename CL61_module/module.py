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
from CL61_module import noise_processing
from CL61_module import classification
from .utils import filename_to_save

plt.style.use('bmh')
COLOR_MAP_NAME = 'cmc.batlow'
COLOR_MAP = cmc.batlow  # type: ignore


class CL61Processor:
    def __init__(self, folder_path, start_datetime=None, end_datetime=None,
                 specific_filename=None, config_path='config_classification.yml'):
        # Path to the data folder
        self.folder_path = folder_path
        # Directory to the module folder 
        self.module_dir = os.path.dirname(__file__)
        # Open config file
        self.config_filepath = config_path
        self.classification_config = self.load_config(config_path)

        self.dataset = None
        self.file_time_mapping = None
        self.period_start = None
        self.period_end = None

        if specific_filename:
            self.load_specific_data(specific_filename)
        else:
            self.load_data_in_timeperiod(start_datetime, end_datetime)

        # TAKES THE SUBCLASSES
        self.plot = Plot(self)
        self.process = Process(self)

    def load_config(self, config_path):
        '''
        Loads a yml config file from given relative or absolute config file path
        Args:
            config_path: relative or absolute config file path
        '''
        # Check if the given config_path is an absolute path
        if os.path.isabs(config_path):
            if os.path.exists(config_path):
                return classification.load_config(config_path)
            else:
                raise ValueError("Config file does not exist: " + config_path)
        else:
            # Join config_path with module_dir
            config_full_path = os.path.join(self.module_dir, config_path)
            if os.path.exists(config_full_path):
                self.config_filepath = config_full_path  # just update the filepath to absolute path
                return classification.load_config(config_full_path)
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

    def load_data_in_timeperiod(self, start_datetime, end_datetime):
        self.period_start = start_datetime
        self.period_end = end_datetime
        self.file_time_mapping = self.get_filepaths_with_timestamps()
        self.dataset = self.load_netcdf4_data_in_timeperiod()

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

        subset = self.dataset.sel(time=slice(start_time, end_time))

        # Create a new CL61Processor instance with the subset data
        subset_processor = CL61Processor(self.folder_path,
                                         start_datetime=start_time,
                                         end_datetime=end_time,
                                         config_path=self.config_filepath)
        
        subset_processor.dataset = subset
        subset_processor.period_start = start_time
        subset_processor.period_end = end_time

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


class Plot:

    def __init__(self, parent: CL61Processor):
        self.CL61_parent = parent

    def colormesh(self,
                       time_period=None,
                       varriable_names=['beta_att', 'linear_depol_ratio'],
                       range_limits=[0, 5000],
                       value_limits=[1e-9, 1e-3],
                       save_fig = False,
                       **plt_kwargs):
        """
        Plots variable(s) in dataset as colormesh

        Args:
            time_period (list, optional): time period to plot only a subset. Defaults to None.
            varriable_names (list, optional): _description_. Defaults to ['beta_att', 'linear_depol_ratio'].
            range_limits (list, optional): _description_. Defaults to [0, 5000].
            value_limits (list, optional): _description_. Defaults to [1e-9, 1e-3].
            save_fig (bool or str, optional): set to True if figure should be saved automatically, else str of file name

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        dataset = self.CL61_parent.dataset

        if dataset is None:
            raise ValueError("No Data has been loaded yet.")

        if isinstance(time_period, list):
            if len(time_period) != 2:
                print(f"Expected 2 values (start time/date, end time/date) but got {len(time_period)}")
                return 0
            else:
                subset = dataset.sel(time=slice(time_period[0], time_period[1]))
        else:
            subset = dataset


        visualization.plot_cl61_as_colormesh(subset,
                                          variable_names=varriable_names,
                                          range_limits=range_limits,
                                          min_value=value_limits[0], max_value=value_limits[1],
                                          color_map=COLOR_MAP,
                                          save_fig = save_fig,
                                          kwargs=plt_kwargs)

        return

    def plot_histogram(self,
                       variable_1='beta_att_clean',
                       variable_2='linear_depol_ratio_clean',
                       classes_variable=None,
                       variable_1_log=True,
                       variable_2_log=False,
                       count_log=False,
                       colormap=COLOR_MAP,
                       save_fig=True):
        """

        Parameters
        ----------
        variable_1
        variable_2
        classes_variable
        variable_1_log
        variable_2_log

        Returns
        -------

        """
        if variable_2 is None:
            visualization.histogram1d(dataset=self.CL61_parent.dataset,
                                      variable_name=variable_1,
                                      hue_variable=classes_variable,
                                      var_transform=('log' if variable_1_log else None),
                                      count_log=count_log,
                                      cmap=colormap,
                                      save_fig=save_fig)
        else:
            visualization.histogram2d(dataset=self.CL61_parent.dataset,
                                      var1 = variable_1,
                                      var2 = variable_2,
                                      log_transforms=[variable_1_log, variable_2_log],
                                      count_tf='log',
                                      min_count_per_bin=2,
                                      cmap=COLOR_MAP,
                                      save_fig=save_fig)

    def plot_classes_colormesh(self,
                               variable_classified='classified_clusters'):
        '''
        Plot classification result in a colormesh 2D plot
        '''
        dataset = self.CL61_parent.dataset

        visualization.plot_classified_colormesh(dataset[variable_classified].T,
                                                dataset['time'],
                                                dataset['range'])
        return

    def vertical_profiles(self, time_of_interest=None,
                          var_names=['beta_att', 'linear_depol_ratio', 'range'],
                          range_limits=[0, 15000],
                          label_first_profile='Beta attenuation',
                          label_second_profile='linear depol ratio',
                          ylabel='range [m]',
                          variables_limits=[[1e-7, 1e-4], [0, 1]],
                          x_scales=['log', 'linear'], ):

        visualization.plotVerticalProfiles(dataset=self.CL61_parent.dataset,
                                           time_period=time_of_interest,
                                           var_names=var_names,
                                           range_limits=range_limits,
                                           xlabel1=label_first_profile,
                                           xlabel2=label_second_profile,
                                           ylabel=ylabel,
                                           title='CL61 profiles',
                                           var_xlims=variables_limits,
                                           x_scales=x_scales,
                                           plot_colors=['#124E63', '#F6A895'])
        return

    def compare_profiles(self, time_period=None, comparison='variable',
                         var_names_1=['beta_att', 'linear_depol_ratio'],
                         var_names_2=['beta_att_clean', 'linear_depol_ratio_clean'],
                         scales=['log', 'lin'],
                         range_limits=[0, 15000],
                         save_fig = True,
                         fig_dpi = 400):
        '''
        Creates 2 subplots to compare side by side vertical profiles of beta attenuation and linear depolarisation ratio.

        Args:
            time_period (str or list of str): time element-s of interest. Expected to be a single str if comparison is not 'time',
            else should be a list of 2 time elements (str) to compare. 
            var_names_1 (list): list of the variables names setting the vertical profiles
            var_names_2 (list): list of the variables names for 2nd profiles for comparison if comparison is 'variable'
        '''
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        if comparison == 'variable':
            axs[0], ax_twin = visualization.plotVerticalProfiles(dataset=self.CL61_parent.dataset,
                                                                 time_period=time_period,
                                                                 var_names=var_names_1, ax=axs[0],
                                                                 range_limits=range_limits)
            axs[1], ax_twin = visualization.plotVerticalProfiles(dataset=self.CL61_parent.dataset,
                                                                 time_period=time_period,
                                                                 var_names=var_names_2, ax=axs[1],
                                                                 range_limits=range_limits)
        elif comparison == 'time':
            if type(time_period) != list:
                raise TypeError(
                    'If vertical profile comparison in time, time_period is expected be a list of 2 time strings')

            axs[0], ax_twin = visualization.plotVerticalProfiles(dataset=self.CL61_parent.dataset,
                                                                 time_period=time_period[0],
                                                                 var_names=var_names_1, ax=axs[0],
                                                                 range_limits=range_limits)
            axs[1], ax_twin = visualization.plotVerticalProfiles(dataset=self.CL61_parent.dataset,
                                                                 time_period=time_period[2],
                                                                 var_names=var_names_1, ax=axs[1],
                                                                 range_limits=range_limits)
        else:
            raise ValueError("'comparison' is expected to be 'variable' or 'time'")
        
        if save_fig:
            filepath = filename_to_save(dataset=self.CL61_parent.dataset.sel(time=time_period, method="nearest"),
                                        save_name=save_fig, suffix='comp_profiles')
            print(f'saved element to {filepath}')
            plt.savefig(filepath, bbox_inches='tight', dpi=fig_dpi)
        
        plt.show()

        return axs
