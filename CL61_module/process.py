# necessary libraries
import numpy as np
import xarray as xr

# function for noise processing and classification
from . import process_noise as p_noise
from . import classification

class ProcessCL61:
    def __init__(self, parent):
        self.CL61_parent = parent
        self.dataset = parent.dataset
        
        self.clustering = classification.ClusteringModule(self)

    def mask_noise(self, beta_att_window_size=[7, 7], linear_depol_windows_size = [1,3]):
        """
        Process noise in the data by searching for negative values for beta_attenuation
        and creates new variables with suffix _clean in the dataset and a 'noise_mask'

        Args:
            beta_att_window_size (int or list of int): size of the window for noise removal for beta attenuation coef
            linear_depol_windows_size (int or list of int): size of the window for noise removal for linear depolarisation ratio
        Returns:
            None
        """
        dataset = self.CL61_parent.dataset
        # Compute non-null masks
        beta_att_non_null_mask = p_noise.non_noise_windows_mask(dataset,
                                                                         variable_name='beta_att',
                                                                         window_size=beta_att_window_size,
                                                                         analysis_type='non-negative')

        linear_depol_non_null_mask = p_noise.non_noise_windows_mask(dataset,
                                                                             variable_name='linear_depol_ratio',
                                                                             window_size=linear_depol_windows_size,
                                                                             analysis_type='range')

        final_mask = beta_att_non_null_mask & linear_depol_non_null_mask
        to_interpolate_mask = beta_att_non_null_mask & ~linear_depol_non_null_mask

        print(
            'The results are stored under new variable: beta_att_clean, linear_depol_ratio_clean, noise mask and to_interpolate_mask')
        
        # Save result into new variable and assign name/description elements
        self.CL61_parent.dataset['beta_att_clean'] = xr.where(final_mask, dataset['beta_att'], np.nan)
        self.CL61_parent.dataset['beta_att_clean'].attrs['units'] = self.dataset['beta_att'].attrs['units']
        self.CL61_parent.dataset['beta_att_clean'].attrs['long_name'] = "filtered attenuated backscatter coefficient"
        self.CL61_parent.dataset['beta_att_clean'].attrs['original_variable'] = 'attenuated backscatter coefficient'

        self.CL61_parent.dataset['linear_depol_ratio_clean'] = xr.where(final_mask, dataset['linear_depol_ratio'], np.nan)
        self.CL61_parent.dataset['linear_depol_ratio_clean'].attrs['long_name'] = "filtered linear depolarisation ratio"
        self.CL61_parent.dataset['beta_att_clean'].attrs['original_variable'] = 'linear depolarisation ratio'

        self.CL61_parent.dataset['noise_mask'] = xr.DataArray(data=final_mask, dims=['time', 'range'])
        self.CL61_parent.dataset['noise_mask'].assign_attrs(name = 'noise mask')

        self.CL61_parent.dataset['to_interpolate_mask'] = xr.DataArray(data=to_interpolate_mask, dims=['time', 'range'])
        self.CL61_parent.dataset['to_interpolate_mask'].assign_attrs(long_name = 'difference in individual noise masks')

        return
    
    def rolling_window_stats(self, variable_name, stat = 'mean',
                             time_window_size=5, range_window_size=5):
        """
        Performs rolling window statistics on dataarray of given variable in dataset.

        Args:
            variable_name (str): variable in dataset
            stat (['mean', 'median', 'std'], optional): statistic. Defaults to 'mean'.
            time_window_size (int, optional): size of window along time. Defaults to 5.
            range_window_size (int, optional): size of window along range. Defaults to 5.

        Raises:
            ValueError: If variable not in dataset

        Returns:
            xarray datarray: datarray of rolling statistics result
        """
        if variable_name not in self.dataset:
            raise ValueError(f"Variable '{variable_name}' not found in the dataset.")

        variable_data = self.dataset[variable_name]

        if stat == 'mean':
            rolling_result = variable_data.rolling(time=time_window_size,
                                                    range=range_window_size,
                                                    min_periods=1, center = True).mean(keep_attrs=True)
        elif stat == 'median':
            rolling_result = variable_data.rolling(time=time_window_size,
                                                    range=range_window_size,
                                                    min_periods=1, center = True).median(keep_attrs=True)
        elif stat == 'std':
            rolling_result = variable_data.rolling(time=time_window_size,
                                        range=range_window_size,
                                        min_periods=1, center = True).std(keep_attrs=True)
        else:
            rolling_result = None
            print(f"stat of type {stat} not supported")
        
        # Save result into array and add relevant attributes
        var_name = f"{variable_name}_roll_{stat}"
        self.dataset[var_name] = rolling_result
        if 'original_variable' in self.dataset['var_name'].attrs.keys():
            self.dataset[var_name].attrs['long_name'] = f"({time_window_size},{range_window_size}) rolling {stat} {self.dataset[var_name].attrs['original_variable']}"
            self.dataset[var_name].attrs['name'] = f"(rolling {stat} {self.dataset[var_name].attrs['original_variable']}"
        elif 'name' in self.dataset['var_name'].attrs.keys():
            self.dataset[var_name].attrs['long_name'] = f"({time_window_size},{range_window_size}) rolling {stat} {self.dataset[var_name].attrs['name']}"
            self.dataset[var_name].attrs['name'] = f"(rolling {stat} {self.dataset[var_name].attrs['name']}"
        
        print(f'Saved the result as variable : {var_name}')

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
        self.CL61_parent.dataset[new_var].attrs['name'] = 'K-mean clusters'
        self.CL61_parent.dataset[new_var].attrs['description'] = f'{cluster_number} K-mean clusters based on the following variables as features: {variable_as_features}'
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
        self.CL61_parent.dataset[classified_cluster_var_name].attrs['name'] = 'classified clusters'
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
        self.CL61_parent.dataset[new_var_name].attrs['name'] = 'point-wise classification'
        print(f" Successful pixel-wise classification stored in dataset under {new_var_name}")
        return