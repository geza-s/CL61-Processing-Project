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
        beta_att_non_null_mask = p_noise.non_noise_windows_mask(dataset,
                                                                         variable_name='beta_att',
                                                                         window_size=window_size,
                                                                         analysis_type='non-negative')

        linear_depol_non_null_mask = p_noise.non_noise_windows_mask(dataset,
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