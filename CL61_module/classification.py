import os

import numpy as np
import xarray as xr
import yaml

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import cmcrameri.cm as cmc

# Needed elements
from .classification_vizalization import visualize_classification_featurespace_2D, visualize_Kmean_results
from .utils import load_config, filename_to_save

# global variables
COLOR_MAP_NAME = 'cmc.batlow'
COLOR_MAP = cmc.batlow # type: ignore because this works
config_classification = None 

def threshold_classify_element(beta_attenuation,
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

def threshold_classify_clusters(dataset,
                                cluster_variable):

    config_classification = load_config()

    class_combination_mapping = config_classification['class_combination_mapping']
    log10_beta_attenuation_thresholds = config_classification['log10_beta_attenuation_thresholds']
    linear_depolarization_thresholds = config_classification['linear_depolarization_thresholds']

    classification_results = {}
    new_classified_array = dataset[cluster_variable].copy()

    for old_cluster_id in np.unique(dataset[cluster_variable]):
        if old_cluster_id == np.nan:
            continue
        mask_cluster = dataset[cluster_variable] == old_cluster_id
        mean_beta = xr.where(mask_cluster, dataset['beta_att_clean'], np.nan).mean(skipna=True) #NA should not be present but making sure
        mean_depol = xr.where(mask_cluster, dataset['linear_depol_ratio_clean'], np.nan).mean(skipna=True) # "

        # Classify the element
        labels = threshold_classify_element(mean_beta, mean_depol,
                                             log10_beta_attenuation_thresholds,
                                             linear_depolarization_thresholds)
        #print(id, mean_beta.values, mean_depol.values, labels)
        if labels == 'Unclassified':
            element_class = 9999
        else:
            element_class = labels[0] + labels[1]
        
        # Map the classification to a corrected combination label
        classification_results[id] = [element_class, class_combination_mapping[element_class]]
        #print(labels, classification_results[id])

        # update new array with new classes
        new_classified_array = xr.where(mask_cluster, element_class, new_classified_array)
    
    return classification_results, new_classified_array

def classify_dataset(dataset,
                      variable_names = ['beta_att_clean','lin_depol_ratio_clean']):
    '''
    Classify based on thresholds directly the whole array elements per elements
    Parameters:
    dataset :
    '''
    # Check for classification config data
    config_classification = load_config()
    log10_beta_attenuation_thresholds = config_classification['log10_beta_attenuation_thresholds']
    linear_depolarization_thresholds = config_classification['linear_depolarization_thresholds']

    # get directly the arrays to classifiy
    beta_attenuation_array = dataset[variable_names[0]]
    lin_depol_array = dataset[variable_names[1]]
    
    # Classify based on beta attenuation
    calssified_result_label = xr.full_like(beta_attenuation_array, np.nan, dtype=np.double)
    for label, (min_value, max_value) in log10_beta_attenuation_thresholds.items():
        mask = (10**min_value <= beta_attenuation_array) & (beta_attenuation_array <= 10**max_value)
        calssified_result_label = xr.where(mask, label, calssified_result_label)

    # Classify based on depolarization
    for label, (min_value, max_value) in linear_depolarization_thresholds.items():
        mask = (min_value <= lin_depol_array) & (lin_depol_array <= max_value)
        calssified_result_label = xr.where(mask, calssified_result_label+label, calssified_result_label)

    return calssified_result_label

def dataset_to_sample_feature(dataset,
                               variable_names=['beta_att_clean', 'linear_depol_ratio_clean', 'range'],
                               transforms=['log', 'lin', 'lin']):
    '''
    Converts variables in xarray dataset of dimension 'time' and 'range' into 1D timeXrange arrays.
    Args:
        - dataset: xarray dataset of dimensions 'time' and 'range
        - variable_names: variables of dataset to extract
        - transforms: Transformation to apply directly when extracted: can be 'log' or else = None
    '''
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

def K_means_classifier(dataset, cluster_N = 8,
                       variable_as_features=['beta_att_clean', 'linear_depol_ratio_clean', 'range'],
                       transforms=['log', 'lin', 'lin'],
                       weights = [1,1,0.5],
                       kmean_method = 'k-means++',
                       plot_result = True,
                       save_fig = True):
    '''
    Implementation of k-mean classification on data from CL61 ceilometer.
    Visualization of results can be called with "plot_result=True".
    
    Parameters:
    - dataset: The input dataset.
    - cluster_N: The number of clusters.
    - variable_as_features: List of variable names to use as features.
    - transforms: List of transformations to apply to the features.
    - weights: List of weights for feature columns.
    - plot_result: If True, visualize the results.
    - kmean_method: The K-means initialization method.
    '''
    # Flatten and trasnform dataset to match needs for clustering
    feature_matrix, nan_rows, cleaned_feature_matrix = dataset_to_sample_feature(dataset=dataset,
                                                                                variable_names=variable_as_features,
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
    original_shape_labels_array = full_shape_labels.reshape(dataset['beta_att_clean'].T.shape)

    if plot_result:
        fig, axes = plt.subplots(1,2, figsize = (15,5), width_ratios=[1, 2])
        fig, axes[0] = visualize_classification_featurespace_2D(feature1_name='log10_beta_attenuation', feature1_flatten=cleaned_feature_matrix[:,0],
                                                                feature2_name='linear_depolarisation_ratio', feature2_flatten=cleaned_feature_matrix[:,1],
                                                                cluster_labels_flatten=cluster_labels,
                                                                fig = fig,
                                                                plot_cbar=False,
                                                                ax = axes[0])
        
        fig, axes[1] = visualize_Kmean_results(dataset,
                                               original_shape_labels_array,
                                               num_categories=cluster_N,
                                               fig = fig,
                                               ax = axes[1])
        if save_fig:
            plt.savefig(filename_to_save(dataset, save_fig))
                
        plt.show()
    return original_shape_labels_array