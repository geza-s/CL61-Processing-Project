# system management
from pathlib import Path
import glob
import os
import importlib

# Array
import xarray as xr
import numpy as np
import datetime as dt
import pandas as pd

# Visualize
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors
import cmcrameri.cm as cmc  # batlow colourmap

# Signal processing
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

plt.style.use('bmh')
COLOR_MAP = 'cmc.batlow'

class CL61Processor:
    def __init__(self, folder_path, start_datetime=None, end_datetime=None, specific_filename = None):
        self.folder_path = folder_path

        if specific_filename:
            specific_filepath = os.path.join(folder_path, specific_filename)
            self.dataset = xr.open_dataset(specific_filepath)
            self.period_start = self.dataset['time'][0].values
            self.period_end  = self.dataset['time'][-1].values
            self.folder_df = None
        else:
            self.period_start = start_datetime
            self.period_end = end_datetime
            self.folder_df = filespaths_in_folder_df()
            self.dataset = load_netcdf4_data_in_timeperiod()

    def filespaths_in_folder_df(self):
        """
        Gets all netcdfs filepaths in the given folder

        Args:
            None
        
        Returns:
            A pandas dataframe with all the file_name_path indexed by datetime
        """
        all_data_files = glob.glob(self.folder_path + '/*.nc')
        get_file_datetime = list(map(lambda date_str: pd.to_datetime(date_str, format='%Y%m%d_%H%M%S'),
                                     [path_name[-18:-3] for path_name in all_data_files]))

        df_data_files = pd.DataFrame([all_data_files, get_file_datetime], index=['file_name_path', 'Datetime']).T
        df_data_files.index = get_file_datetime
        df_data_files = df_data_files.sort_index()
        return df_data_files

    def load_netcdf4_data_in_timeperiod(self):
        '''
        Load the netcdf4 files in the folder and inside the given period
        '''
        selected_data = self.folder_df[self.folder_df['Datetime'].between(self.period_start, self.period_end)]
        return xr.open_mfdataset(selected_data['file_name_path'], chunks={'time': 300})

    def slice_data(self, start_time, end_time):
        """
        Slice the loaded data for a specified time range.

        Args:
            start_time (str): Start time (e.g., 'YYYY-MM-DD').
            end_time (str): End time (e.g., 'YYYY-MM-DD').

        Returns:
            None
        """
        # Implement data slicing logic based on time

    def reload_modules(self):
        importlib.reload(visualization)
        importlib.reload(noise_processing)
        return
    
    def mask_noise(self, window_size=[7,7]):
        """
        Process noise in the data by searching for negative values for beta_attenuation
        and creates new variables with suffix _clean in the dataset and a 'noise_mask'

        Args:
            window_size (int or list of int): size of the window for noise removal padding around null value.
        Returns:
            None
        """
        # Compute non-null masks
        beta_att_non_null_mask = noise_processing.non_noise_windows_mask_v2(self.dataset,
                                                                             variable_name='beta_att',
                                                                             window_size=window_size,
                                                                             analysis_type='non-negative')
        
        linear_depol_non_null_mask= noise_processing.non_noise_windows_mask_v2(self.dataset,
                                                                                variable_name='linear_depol_ratio',
                                                                                window_size=[1, 3],
                                                                                analysis_type='range')

        # Compute final arrays as the INTERSECTION of both masks (biggest common pixels array)
        #final_mask = np.logical_and(null_mask_beta_att, null_mask_lin)
        #to_interpolate_mask = ((null_mask_beta_att.astype(int)-null_mask_lin.astype(int))==1)
        final_mask = beta_att_non_null_mask & linear_depol_non_null_mask
        to_interpolate_mask = beta_att_non_null_mask & ~linear_depol_non_null_mask
        
        # Store new arrays
        #self.dataset['beta_att_clean'] = xr.where(final_mask.T, self.dataset['beta_att'], np.nan)
        #self.dataset['linear_depol_ratio_clean'] = xr.where(final_mask.T, self.dataset['linear_depol_ratio'], np.nan)
        #self.dataset['noise_mask'] = xr.DataArray(data=final_mask.T, dims=['time', 'range'])
        #self.dataset['to_interpolate_mask'] = xr.DataArray(data=to_interpolate_mask.T, dims=['time', 'range'])
        
        self.dataset['beta_att_clean'] = xr.where(final_mask, self.dataset['beta_att'], np.nan)
        self.dataset['linear_depol_ratio_clean'] = xr.where(final_mask, self.dataset['linear_depol_ratio'], np.nan)
        self.dataset['noise_mask'] = xr.DataArray(data=final_mask, dims=['time', 'range'])
        self.dataset['to_interpolate_mask'] = xr.DataArray(data=to_interpolate_mask, dims=['time', 'range'])

        return
    
    
    def classify_data_kmean(self, variable_as_features=['beta_att_clean', 'linear_depol_ratio_clean'],
                            cluster_number = 8, plot_result = True):
        """
        Perform data classification by k-means.

        Returns:
            None
        """
        # Implement classification logic
        classification_result_array = classification.K_means_classifier(dataset=self.dataset, cluster_N=cluster_number,
                                                                        variable_names=variable_as_features)
        
        self.dataset['kmean_classified'] = xr.DataArray(data=classification_result_array.T, dims=['time', 'range'])
        
        if plot_result:
            visualization.plot_classifiction_result(dataset=self.dataset, classified_var_name = 'kmean_classified')
        
        return
        
    def vertical_profiles(self, time_of_interest = None,
                          var_names = ['beta_att', 'linear_depol_ratio', 'range'],
                          hlims = [0, 15000],
                          label_first_profile ='Beta attenuation',
                          label_second_profile = 'linear depol ratio',
                          ylabel = 'range [m]',
                          variables_limits = [[1e-7, 1e-4],[0,1]],
                          x_scales = ['log', 'linear'],):
        
        

        visualization.plotVerticalProfiles(dataset=self.dataset, time_period = time_of_interest,
                                            var_names = var_names,
                                            hlims = hlims,
                                            xlabel1= label_first_profile,
                                            xlabel2 = label_second_profile,
                                            ylabel = ylabel,
                                            title = 'CL61 profiles',
                                            var_xlims = variables_limits,
                                            x_scales = x_scales,
                                            plot_colors = ['#124E63', '#F6A895'])
        return

    def compare_profiles(self, time_period = None,
                          var_names_1 = ['beta_att', 'linear_depol_ratio', 'range'],
                          var_names_2 = ['beta_att_clean', 'linear_depol_ratio_clean', 'range'],
                          hlims = [0,15000]):
        
        fig, axs = plt.subplots(1, 2, figsize=(10,6))
        axs[0], ax_twin = visualization.plotVerticalProfiles(dataset=self.dataset, time_period=time_period,
                                                     var_names=var_names_1, ax=axs[0],
                                                     hlims=hlims)
        axs[1], ax_twin = visualization.plotVerticalProfiles(dataset=self.dataset, time_period=time_period,
                                                     var_names=var_names_2, ax=axs[1],
                                                     hlims=hlims)
        plt.show()
        
        return axs

    def visualize_data(self, plot_type='profile', time_period = None,
                        varriable_names = ['beta_att', 'linear_depol_ratio'],
                        range_limits = [0, 5000],
                        value_limits = [1e-9, 1e-3]):
        """
        Visualize the processed data.

        Args:
            plot_type (str): Type of plot to generate ('profile', 'colormesh')

        Returns:
            None
        """
        if self.dataset is None:
            raise ValueError("No Data has been loaded yet.")

        if time_period:
            if type(time_period) == str:
                subset = self.dataset.sel(time = time_period, method="nearest")
            elif type(time_period) == list:
                if len(time_period)>2:
                    print(f"Expected max 2 values (start time/date, end time/date) but got {len(time_period)}")
                    return 0
                else:
                    subset = self.dataset.sel(time=slice(time_period[0], time_period[1]))
        else:
            subset = self.dataset

        if plot_type == 'profile':
            visualization.plotCl61asProfile(subset, time_period=None,
                                      variable=varriable_names, hlims=range_limits,
                                      color_map=COLOR_MAP)
        elif plot_type == 'colormesh':
            visualization.plotCL61AsColomersh(subset, variable_names=varriable_names,
                                       hlims=range_limits,
                                       min_value=value_limits[0], max_value = value_limits[1],
                                       color_map=COLOR_MAP)
        else:
            raise ValueError(f"Unsupported plot type: {plot_type}")

    def close(self):
        """
        Close the NetCDF dataset.

        Returns:
            None
        """
        self.dataset.close()


# Example usage:
#if __name__ == "__main__":
#    processor = CL61Processor(folder_path='../SIE-Project/CL61/Code', specific_filepath='temp_20230204.nc')
#    processor.mask_noise()
#    processor.classify_data()
#    processor.visualize_data(plot_type='colormesh')
#    processor.close()
