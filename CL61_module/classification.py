import numpy as np
import pandas as pd
import xarray as xr

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import datashader as ds
import datashader.transfer_functions as tf
import cmcrameri.cm as cmc

COLOR_MAP_NAME = 'cmc.batlow'
COLOR_MAP = cmc.batlow

def visualize_classification_featurespace_2D(feature1_flatten, feature2_flatten, cluster_labels, cluster_N):

    # Assuming 'cleaned_feature_matrix' is your feature matrix
    df = pd.DataFrame({'log10_Beta_att': feature1_flatten,
                    'Linear_depol': feature2_flatten,
                    'Cluster_Labels': cluster_labels})

    cvs = ds.Canvas(plot_width=500, plot_height=500)
    agg = cvs.points(df, 'log10_Beta_att', 'Linear_depol', ds.mean('Cluster_Labels'))

    # Use the color map you defined earlier
    #colors_rgb = ()
    #color_key = dict(enumerate(colors_rgb))

    # Create a custom colormap in a format that Datashader can use
    #cmap = tf.color_map(builders=[colors_rgb.tolist()], name="custom_cmap")

    img = tf.shade(agg, cmap=COLOR_MAP)

    # Display the plot
    tf.set_background(img, 'black')

    return img 

def visualize_Kmean_results(dataset, original_shape_labels_array, num_categories):
    # Define the original colormap (e.g., 'viridis')
    original_cmap = plt.get_cmap(COLOR_MAP_NAME)

    # Define the number of discrete categories
    #num_categories = np.unique(original_shape_labels_array).size  # Adjust as needed

    # Create a list of evenly spaced values to sample the colormap
    color_values = np.linspace(0, 1, num_categories)

    # Sample the original colormap at the specified values
    discrete_colors = original_cmap(color_values)

    # Create a custom ListedColormap with the discrete colors
    discrete_cmap = ListedColormap(discrete_colors)

    plt.figure(figsize=(8, 5))  # Set the figure size as needed

    # Create a colored mesh plot using the custom colormap
    plot = plt.pcolormesh(dataset['time'], dataset['range'], original_shape_labels_array,
                cmap=discrete_cmap, vmin=0, vmax=num_categories - 1)

    # Add a colorbar with discrete color labels
    norm = plt.Normalize(vmin=0, vmax=num_categories - 1)
    cbar = plt.colorbar(cmap=discrete_cmap, ticks=range(num_categories))
    cbar.set_ticklabels(range(num_categories))  # Set labels to category indices

    # Set labels for x and y axes (if needed)
    plt.ylim([0,5000])
    plt.xlabel('Range')
    plt.ylabel('Time')

    # Set the title of the plot (if needed)
    plt.title('K-Means Clustering Results')

    # Show the plot
    plt.show()

    return

def dataset_to_sample_feature(dataset, variable_names=['beta_att_clean', 'linear_depol_ratio_clean', 'range'], transforms=['log', 'lin', 'lin']):
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
                       plot_result = True):
    feature_matrix, nan_rows, cleaned_feature_matrix = dataset_to_sample_feature(dataset=dataset,
                                                                                variable_names=variable_as_features)

    # Try with standardisation
    scaler = StandardScaler()
    feature_matrix_standardized = scaler.fit_transform(cleaned_feature_matrix)
    
    print(f'Scaler means: {scaler.mean_}  \n and scales: {scaler.scale_}')

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=cluster_N, init='k-means++', n_init='auto', random_state=0, tol=1e-5)
    cluster_labels = kmeans.fit_predict(feature_matrix_standardized)
    
    # Create an array to store cluster labels with NaN rows
    full_shape_labels = np.full((feature_matrix.shape[0],), np.nan)

    # Fill in the cluster labels for rows without NaN values
    full_shape_labels[~nan_rows] = cluster_labels

    # Reshape the labels to the original shape (time x range)
    original_shape_labels_array = full_shape_labels.reshape(dataset['beta_att_clean'].T.shape)

    if plot_result:
        visualize_classification_featurespace_2D(feature1_flatten=feature_matrix_standardized[:,0],
                                                 feature2_flatten=feature_matrix_standardized[:,1],
                                                 cluster_labels=cluster_labels,
                                                 cluster_N=cluster_N)
        
        visualize_Kmean_results(dataset, original_shape_labels_array, num_categories=cluster_N)

    return original_shape_labels_array

# ------------------------------------------------------------------------------------------------

# Define the classification thresholds
linear_depolarization_thresholds = {
    0 : [0, 0.2], # 'liquid_droplets
    1 : [0.2, 0.35], # 'snow'
    2 : [0.35, 0.5], # 'ice_crystals'
    3 : [0.5, 1] #'graupel'
}

beta_attenuation_thresholds = {
    0 : [1e-10, 1e-6], # 'clear'
    1 : [1e-6, 1e-5], # 'low density'
    5 : [1e-5, 5e-5], # 'precipitation'
    9 : [5e-5, 1e-3] # 'clouds
}

class_combination_mapping = {
'clear & clear': 0,
'low density & liquid_droplets': 1,
'low density & snow': 2,
'low density & ice_crystals': 3,
'low density & graupel': 4,
'precipitation & liquid_droplets': 5,
'precipitation & snow': 6,
'precipitation & ice_crystals': 7,
'precipitation & graupel': 8,
'clouds & liquid_droplets': 9,
'clouds & snow': 10,
'clouds & ice_crystals': 11,
'clouds & graupel': 12
}

category_colors = [
    'xkcd:light grey',       # clear & clear
    'lightskyblue',          # low density & liquid_droplets
    'xkcd:cream',            # low density & snow
    'xkcd:pale lilac',       # low density & ice_crystals
    'xkcd:wheat',            # low density & graupel
    'xkcd:light cyan',       # precipitation & liquid_droplets
    'xkcd:light pink',       # precipitation & snow
    'xkcd:light lilac',     # precipitation & ice_crystals
    'xkcd:golden',           # precipitation & graupel
    'xkcd:prussian blue',    # clouds & liquid_droplets
    'xkcd:grey purple',      # clouds & snow
    'xkcd:dark mauve',       # clouds & ice_crystals
    'gray'       # clouds & graupel
]

def threshold_classify_element(beta_attenuation, depolarization,
                                beta_attenuation_thresholds = beta_attenuation_thresholds,
                                depolarization_thresholds = linear_depolarization_thresholds):
  
        # Classify based on beta attenuation
    beta_attenuation_class = None
    for label, (min_value, max_value) in beta_attenuation_thresholds.items():
        if min_value <= beta_attenuation <= max_value:
            beta_attenuation_class = label
            break
    
    if beta_attenuation_class == 'clear':
        return [beta_attenuation_class, 'clear']
    else: 
        # Classify based on depolarization
        depolarization_class = None
        for label, (min_value, max_value) in depolarization_thresholds.items():
            if min_value <= depolarization <= max_value:
                depolarization_class = label
                break
    
    # Return the classification
    if depolarization_class is not None and beta_attenuation_class is not None:
        return [beta_attenuation_class, depolarization_class]
    else:
        return 'Unclassified'


def threshold_classify_kmean_result(dataset, original_shape_labels_array,
                                    class_combination_mapping = class_combination_mapping ):

    classification_results = {}

    for id in np.unique(original_shape_labels_array):
        if id == np.nan:
            continue
        mask_i = original_shape_labels_array == id
        mean_beta = xr.where(mask_i.T, ds['beta_att_clean'], np.nan).mean(skipna=True)
        mean_depol = xr.where(mask_i.T, ds['linear_depol_ratio_clean'], np.nan).mean(skipna=True)
        
        # Classify the element
        labels = threshold_classify_element(mean_beta, mean_depol)
        #print(id, mean_beta.values, mean_depol.values, labels)
        if labels == 'Unclassified':
            element_class = 'Unclassified'
        else:
            element_class = f'{labels[0]} & {labels[1]}'
        
        # Map the classification to a corrected combination label
        combination_class = class_combination_mapping.get(element_class, np.nan)
        
        classification_results[id] = combination_class

    final_classification_original_shape = np.copy(original_shape_labels_array)
    for old_class, new_class in classification_results.items():
        mask_old_class = original_shape_labels_array==old_class
        final_classification_original_shape[mask_old_class] = new_class
    
    return classification_results, final_classification_original_shape

def classify_array(beta_attenuation, depolarization,
                                beta_attenuation_thresholds = beta_attenuation_thresholds,
                                depolarization_thresholds = linear_depolarization_thresholds):
    # Classify based on beta attenuation
    calssified_result_label = np.empty(beta_attenuation.shape)
    for label, (min_value, max_value) in beta_attenuation_thresholds.items():
        mask = (min_value <= beta_attenuation) & (beta_attenuation <= max_value)
        calssified_result_label[mask] = label

    # Classify based on depolarization
    for label, (min_value, max_value) in depolarization_thresholds.items():
        mask = (min_value <= depolarization) & (depolarization <= max_value)
        calssified_result_label[mask] += label

    calssified_result_label[np.isnan(beta_attenuation)] = np.nan

    return calssified_result_label

from matplotlib.colors import ListedColormap

def plot_classified_colormesh(classified_array, 
                              time_array,
                              range_array,
                              num_categories = 13,
                              category_colors = category_colors,
                              class_combination_mapping = class_combination_mapping):

    my_cmap = ListedColormap(category_colors)

    plt.figure(figsize=(10, 5))  # Set the figure size as needed

    # Create a colored mesh plot using the custom colormap
    plot = plt.pcolormesh(time_array, range_array, classified_array,
                cmap=my_cmap, vmin=0, vmax=num_categories)

    # Add a colorbar with discrete color labels
    norm = plt.Normalize(vmin=0, vmax=num_categories)
    cbar = plt.colorbar(cmap=my_cmap, ticks=range(num_categories))
    cbar.set_ticklabels(list(class_combination_mapping.keys()))  # Set labels to category indices

    # Set labels for x and y axes (if needed)
    plt.ylim([0,5000])
    plt.xlabel('Range')
    plt.ylabel('Time')

    # Set the title of the plot (if needed)
    plt.title('Classification Results')

    # Show the plot
    plt.show()
    return