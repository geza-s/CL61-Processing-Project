import os

# Dataset Treatment
import numpy as np
import xarray as xr
import yaml

# Unsupervised clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Visualization
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

# Import associated functions
from .classification_vizalization import visualize_classification_featurespace_2D, visualize_cluster_results, plot_classified_timeserie
from .utils import load_config, filename_to_save

# global variables
COLOR_MAP_NAME = 'cmc.batlow'
COLOR_MAP = cmc.batlow # type: ignore because this works
config_classification = None 

class CL61Classifier:
    def __init__(self, parent, dataset = None, config_filepath = None):
        if dataset:
            # To test this class
            self.dataset = dataset
            self.parent = None
            self.config = load_config(config_filepath)
        else:
            # Usual call in CL61 module
            self.dataset = parent.dataset
            self.parent = parent
            # Config elements 
            self.config = self._load_config(parent.config_path)

        # Some variables settings: name of the variables
        self.cluster_result_var = {'KMEAN' : 'kmean_clusters', 'DBSCAN':'dbscan_clusters'}
        
    def _load_config(self, config_path):
        '''
        Loads a yml config file from given relative or absolute config file path
        Args:
            config_path: relative or absolute config file path
        '''
        # Check if the given config_path is an absolute path
        if os.path.isabs(config_path):
            if os.path.exists(config_path):
                self.config_path = config_path
                return load_config(config_path)
            else:
                raise ValueError("Config file does not exist: " + config_path)
        else:
            # Join config_path with module_dir
            config_full_path = os.path.join(self.parent.module_dir, config_path)
            if os.path.exists(config_full_path):
                self.config_path = config_full_path # just update the filepath to absolute path
                return load_config(config_full_path)
            else:
                raise ValueError("Config file does not exist in module_dir: " + config_full_path)

    def dataset_to_sample_feature(self,
                               variable_names=['beta_att_clean', 'linear_depol_ratio_clean', 'range'],
                               transforms=['log', 'lin', 'lin']):
        '''
        Converts variables in xarray dataset of dimension 'time' and 'range' into 1D timeXrange arrays.
        Args:
            - dataset: xarray dataset of dimensions 'time' and 'range
            - variable_names: variables of dataset to extract
            - transforms: Transformation to apply directly when extracted: can be 'log' or else = None
        Returns:
            [feature_matrix_with_Nans, nan_rows_mask, feature_matrix_clean] : (Numpy ndarrays) 
                First the whole feature matrix with also nan values, then the mask of nan rows, and the final feature matrix
        '''
        # Get ref for dataset
        dataset = self.dataset
        
        # Create an empty list to store transformed feature arrays
        feature_arrays = []
        array_size = dataset['time'].size*dataset['range'].size

        for variable_name, transform in zip(variable_names, transforms):
            feature_array = dataset[variable_name].T.values

            # Handle the 'log' transformation
            if transform == 'log':
                if np.any(feature_array < 0):
                    # Handles negative or zero values by adding a small offset (1e-10) before taking the logarithm
                    print(f"Warning: Found negative or zero values in '{variable_name}'. Log transformation may result in NaN.")
                    feature_array = np.log10(np.maximum(feature_array, 1e-10))
                else:
                    feature_array = np.log10(feature_array)

            if feature_array.size != array_size:
                feature_array = np.repeat(feature_array, array_size//feature_array.size)

            feature_arrays.append(feature_array.flatten())

        # Stack the feature arrays vertically to create a feature matrix
        original_feature_matrix = np.column_stack(feature_arrays)
        print(original_feature_matrix.shape)

        # Remove rows with NaN values
        nan_rows = np.isnan(original_feature_matrix).any(axis=1)
        cleaned_feature_matrix = original_feature_matrix[~nan_rows]

        return original_feature_matrix, nan_rows, cleaned_feature_matrix

    def Kmeans_clustering(self, cluster_N=8,
                            variable_as_features=['beta_att_clean', 'linear_depol_ratio_clean', 'range'],
                            transforms=['log', 'lin', 'lin'],
                            weights=[1, 1, 0.25],
                            kmean_method='k-means++',
                            plot_result=True,
                            save_fig=True,
                            plot_range=[0, 10000]):
        '''
        Implementation of k-mean classification on data from CL61 ceilometer.
        Visualization of results can be called with "plot_result=True".

        Parameters:
        - cluster_N: The number of clusters.
        - variable_as_features: List of variable names to use as features.
        - transforms: List of transformations to apply to the features.
        - weights: List of weights for feature columns.
        - plot_result: If True, visualize the results.
        - kmean_method: The K-means initialization method.
        '''

        # Flatten and transform dataset to match needs for clustering
        feature_matrix, nan_rows, cleaned_feature_matrix = self.dataset_to_sample_feature(variable_names=variable_as_features,
                                                                                          transforms=transforms)

        # Standardize the feature matrix
        scaler = StandardScaler()
        feature_matrix_standardized = scaler.fit_transform(cleaned_feature_matrix)
        print(f'Scaler means: {scaler.mean_}  \n and scales: {scaler.scale_}')

        if weights is not None:
            if len(weights) == 1:
                weights = weights * np.ones(len(variable_as_features))
            elif len(weights) != len(variable_as_features):
                raise ValueError("Weights should be None or the wished weight of each variable")

            # Weighting as 1/weight to increase the distance
            for i in range(len(weights)):
                feature_matrix_standardized[:, i] *= weights[i]

        # Perform k-means clustering
        kmeans = KMeans(n_clusters=cluster_N, init=kmean_method, n_init='auto')
        cluster_labels = kmeans.fit_predict(feature_matrix_standardized)

        # Create an array to store cluster labels with NaN rows
        full_shape_labels = np.full((feature_matrix.shape[0],), np.nan)

        # Fill in the cluster labels for rows without NaN values
        full_shape_labels[~nan_rows] = cluster_labels

        # Reshape the labels to the original shape (time x range)
        original_shape_labels_array = full_shape_labels.reshape(self.dataset['beta_att_clean'].T.shape)

        if plot_result:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5), width_ratios=[1, 2])
            fig, axes[0] = visualize_classification_featurespace_2D(feature1_name='log10_beta_attenuation',
                                                                    feature1_flatten=cleaned_feature_matrix[:, 0],
                                                                    feature2_name='linear_depolarisation_ratio',
                                                                    feature2_flatten=cleaned_feature_matrix[:, 1],
                                                                    cluster_labels_flatten=cluster_labels,
                                                                    fig=fig,
                                                                    plot_cbar=False,
                                                                    ax=axes[0]
                                                                    )
            fig, axes[1] = visualize_cluster_results(dataset= self.dataset,
                                                    original_shape_labels_array=original_shape_labels_array,
                                                    num_categories=cluster_N,
                                                    fig=fig,
                                                    ax=axes[1],
                                                    range_limits=plot_range
                                                    )
            if save_fig:
                plt.savefig(filename_to_save(self.dataset, save_fig, suffix='KMean'))

            plt.show()
        
        # Save clustering results to the dataset
        new_var = self.cluster_result_var['KMEAN']
        print(f'saving results under following{new_var}')
        self.dataset[new_var] = xr.DataArray(data=original_shape_labels_array.T, dims=['time', 'range'])
        self.dataset[new_var].attrs['name'] = 'K-mean clusters'
        self.dataset[new_var].attrs['description'] = f'{cluster_N} K-mean clusters based on the following variables as features: {variable_as_features}'

        return 

    def dbscan_clustering(self, variable_as_features=['beta_att', 'linear_depol_ratio', 'range'],
                            transforms = ['log', 'lin', 'lin'],
                            weights=[1, 1, 0.3],
                            dbscan_eps=0.5, dbscan_min_samples=5,
                            plot_result=True, plot_range = [0, 15000], save_fig=True):
        """
        Perform DBSCAN clustering on the dataset.

        Parameters:
        - variable_as_features: List of variable names to use as features.
        - weights: List of weights for feature columns.
        - dbscan_eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        - dbscan_min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
        - plot_result: If True, visualize the results.
        - save_fig: If True, figure plotted should be saved. If str, figure saved with name as given.

        Returns:
        - None
        """
        # Reference to dataset for easier use
        dataset = self.dataset
        
        # Flatten and trasnform dataset to match needs for clustering
        feature_matrix, nan_rows, cleaned_feature_matrix = self.dataset_to_sample_feature(variable_names=variable_as_features,
                                                                                          transforms=transforms)

        # Standardize the feature matrix
        scaler = StandardScaler()
        feature_matrix_standardized = scaler.fit_transform(cleaned_feature_matrix)
        print(f'Scaler means: {scaler.mean_}  \n and scales: {scaler.scale_}')

        # Apply weights
        if weights is not None:
            if len(weights) == 1:
                weights = weights * np.ones(len(variable_as_features))
            elif len(weights) != len(variable_as_features):
                raise ValueError("Weights should be None or the wished weight of each variable")
            
            # Weighting as 1/weight to increase the distance 
            for i in range(len(weights)):
                feature_matrix_standardized[:, i] *= weights[i]

        # Remove row from feature matrix with nan-values
        feature_matrix[nan_rows]
        
        # Create an array to store cluster labels with NaN rows
        full_shape_labels = np.full((feature_matrix.shape[0],), np.nan)
        
        # Implement DBSCAN clustering logic
        db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        cluster_labels = db.fit_predict(feature_matrix_standardized)

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise_ = list(cluster_labels).count(-1)

        #To avoid -1 cluster as "outliers" -> we bring noise cluster to No 0.
        cluster_labels += 1 
        
        # Fill in the cluster labels for rows without NaN values
        full_shape_labels[~nan_rows] = cluster_labels

        # Reshape the labels to the original shape (time x range)
        original_shape_labels_array = full_shape_labels.reshape(dataset['beta_att_clean'].T.shape)

        # Save clustering results to the dataset
        new_var = self.cluster_result_var['DBSCAN']
        print(f'saving DBSCAN clustering results under following variable {new_var}')
        self.dataset[new_var] = xr.DataArray(data=original_shape_labels_array.T, dims=['time', 'range'])
        self.dataset[new_var].attrs['name'] = 'DBSCAN clusters'
        self.dataset[new_var].attrs['description'] = f'DBSCAN clustering with eps={dbscan_eps}, min_samples={dbscan_min_samples}'
        
        if plot_result:
            fig, axes = plt.subplots(1,2, figsize = (15,5), width_ratios=[1, 2])
            fig, axes[0] = visualize_classification_featurespace_2D(feature1_name='log10_beta_attenuation',
                                                                    feature1_flatten=cleaned_feature_matrix[:,0],
                                                                    feature2_name='linear_depolarisation_ratio',
                                                                    feature2_flatten=cleaned_feature_matrix[:,1],
                                                                    cluster_labels_flatten=cluster_labels,
                                                                    fig = fig,
                                                                    plot_cbar=False,
                                                                    ax = axes[0])
            
            fig, axes[1] = visualize_cluster_results(dataset= self.dataset,
                                                    original_shape_labels_array=original_shape_labels_array,
                                                    num_categories=n_clusters_+1,
                                                    fig=fig,
                                                    ax=axes[1],
                                                    range_limits=plot_range
                                                    )
            
            if save_fig:
                plt.savefig(filename_to_save(dataset, save_fig, suffix='DBScan'))
                    
            plt.show()
        
        return

    def classify_clusters(self,
                            cluster_variable='kmean_clusters',
                            beta_attenuation_varname='beta_att_clean',
                            linear_depol_ratio_varname = 'linear_depol_ratio_clean',
                            verbose = False):
        '''
        Classifies each cluster of cluster_variable array based on determined thresholds
        '''
        print(f'Classifying cluster from {cluster_variable}')

        # Array for classified results
        new_classified_array = self.dataset[cluster_variable].copy()

        for cluster_id in np.unique(self.dataset[cluster_variable]):
            
            if cluster_id == np.nan:
                continue
            
            # Get all elements related to cluster
            mask_cluster = self.dataset[cluster_variable] == cluster_id

            # Get mean measures of cluster
            mean_beta = xr.where(mask_cluster, self.dataset[beta_attenuation_varname], np.nan).mean(skipna=True) #NA should not be present but making sure
            mean_depol = xr.where(mask_cluster, self.dataset[linear_depol_ratio_varname], np.nan).mean(skipna=True) # same

            # Classify the element
            cluster_class = threshold_classify_element(self.config, mean_beta, mean_depol)
            if cluster_class == 'Unclassified':
                element_class = np.NaN
            else:
                element_class = cluster_class['class_id']

            if verbose:
                print(id, mean_beta.values, mean_depol.values, cluster_class)
            
            # update new array with new classes
            new_classified_array = xr.where(mask_cluster, element_class, new_classified_array)

        classified_cluster_var_name = 'classified_clusters'
        self.dataset[classified_cluster_var_name] = new_classified_array
        self.dataset[classified_cluster_var_name].attrs['name'] = 'classified clusters'
        print(f" Successful cluster classification stored in dataset under {classified_cluster_var_name}")

        return

    def classify_elementwise(self,
                             beta_attenuation_varname='beta_att_clean',
                             linear_depol_ratio_varname = 'linear_depol_ratio_clean'):
        '''
        Classifies directly element wise based on thresholds focused on the variables in question
        '''

        # get directly the arrays to classifiy
        beta_attenuation_array = self.dataset[beta_attenuation_varname]
        lin_depol_array = self.dataset[linear_depol_ratio_varname]
        
        # New array for cluster results
        classified_result_label = xr.full_like(beta_attenuation_array, np.nan)

        # Classify based on ranges as defined in config file
        for category in self.config['classes']:
            beta_range = category['beta_attenuation_range']
            depolarization_range = category['linear_depolarisation_ratio_range']
            cluster_id = category['class_id']

            mask_beta_att = (beta_range[0] <= beta_attenuation_array) & (beta_attenuation_array <= beta_range[1])
            mask_ldr = (depolarization_range[0] <= lin_depol_array) & (lin_depol_array <= depolarization_range[1])
            common_mask = mask_beta_att & mask_ldr
            classified_result_label = xr.where(common_mask, cluster_id, classified_result_label)

        new_var_name = 'classified_elements'
        self.dataset[new_var_name] = classified_result_label

        self.dataset[new_var_name].attrs['name'] = 'point-wise classification'
        print(f"Successful pixel-wise classification stored in dataset under {new_var_name}")
        return

def threshold_classify_element_old(beta_attenuation,
                                depolarization,
                                log10_beta_attenuation_thresholds,
                                linear_depolarization_thresholds):

    # Classify based on beta attenuation
    beta_attenuation_class = None
    for label, (min_value, max_value) in log10_beta_attenuation_thresholds.items():
        if 10**min_value <= beta_attenuation <= 10**max_value:
            beta_attenuation_class = label
            break
    
    if beta_attenuation_class == 'clear':
        return [beta_attenuation_class, 'clear']
    else: 
        # Classify based on depolarization
        depolarization_class = None
        for label, (min_value, max_value) in linear_depolarization_thresholds.items():
            if min_value <= depolarization <= max_value:
                depolarization_class = label
                break
    
    # Return the classification
    if depolarization_class is not None and beta_attenuation_class is not None:
        return [beta_attenuation_class, depolarization_class]
    else:
        return 'Unclassified'

def threshold_classify_element(class_config, beta_attenuation, linear_depolarization_ratio):
    '''finds based on pair of measure (beta attenuation, linear depolarisation ratio) the class related to it
    returns:
        dictionary of the related class as defined in the config file
    '''
    for category in class_config['classes']:
        beta_range = category['beta_attenuation_range']
        depolarization_range = category['linear_depolarisation_ratio_range']

        if beta_range[0] <= beta_attenuation <= beta_range[1] and \
           depolarization_range[0] <= linear_depolarization_ratio <= depolarization_range[1]:
            return category

    return 'Unclassified'