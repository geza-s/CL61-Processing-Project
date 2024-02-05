# array management
import numpy as np
import pandas as pd

# Basic plots
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap

# batlow colormap
import cmcrameri.cm as cmc

# Needed function
from scipy.stats import mode  # for hexbins better plotting

# Extra function
from CL61_module.utils import filename_to_save

# SET UP
COLOR_MAP_NAME = 'cmc.batlow'
COLOR_MAP = cmc.batlow  # type: ignore
COLOR_CODES_BLUE_YEL = ['#03245C', '#D69444']


def visualize_classification_featurespace_2D(log10_beta_att_flatten,
                                             linear_depol_ratio_flatten,
                                             cluster_labels_flatten,
                                             fig=None, ax=None,
                                             plot_cbar=True,
                                             cmap=COLOR_MAP):
    """
    Plots the classification results in the 2D feature space of log10 attenuated backscatter and linear depolarisation
    ratio.

    Args:
        log10_beta_att_flatten (numpy array): Flattened numpy array of log 10 backscatter attenuation coefficients
        linear_depol_ratio_flatten (numpy array): Flattened array of linear depolarisation ratios
        cluster_labels_flatten (numpy array): Flattened array of cluster ids
        fig (matplotlib figure, optional): Figure to plot onto. Defaults to None.
        ax (matplotlib axe, optional): Axe to plot onto. Defaults to None.
        plot_cbar (bool, optional): To plot a colorbar or not. Defaults to True.
        use_datashader (bool, optional): _description_. Defaults to False.
        cmap (_type_, optional): _description_. Defaults to COLOR_MAP.

    Returns:
        fig, ax: figure and ax of the plot
    """

    # Assuming 'cleaned_feature_matrix' is your feature matrix
    df = pd.DataFrame({'log10_beta': log10_beta_att_flatten,
                       'LDR': linear_depol_ratio_flatten,
                       'cluster_labels': cluster_labels_flatten})

    # cvs = ds.Canvas(plot_width=500, plot_height=500)
    if (ax is None) | (fig is None):
        print("new figure")  # debugging
        set_figure = True
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        set_figure = False

    # Define the original colormap (e.g., 'viridis')
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Define the number of discrete categories
    num_categories = np.unique(df['cluster_labels']).size

    # Create a list of evenly spaced values to sample the colormap
    color_values = np.linspace(0, 1, num_categories)

    # Sample the original colormap at the specified values
    discrete_colors = cmap(color_values)

    # Create a custom ListedColormap with the discrete colors
    discrete_cmap = ListedColormap(discrete_colors)

    # Function choosing the most common value in bin (hexbin) for associated color
    def most_freq_value(a):
        return np.bincount(a).argmax()

    # Hexbin
    artist = ax.hexbin(log10_beta_att_flatten, linear_depol_ratio_flatten, C=cluster_labels_flatten,
                       gridsize=500, cmap=discrete_cmap,
                       reduce_C_function=most_freq_value, mincnt=2)

    if plot_cbar:
        fig.colorbar(artist, ax=ax, orientation='vertical')

    # ax.set_title('Clustering in Feature space')
    ax.set_xlabel('$Log_{10}$ attenuated backscatter \n coefficient [$m^{-1}~sr^{-1}$]')
    ax.set_ylabel('Linear depolarization ratio')

    #if set_figure:
    #    plt.show()

    return fig, ax


def visualize_cluster_results(dataset,
                              original_shape_labels_array,
                              num_categories,
                              fig=None,
                              ax=None,
                              range_limits=[0, 5000],
                              color_map=COLOR_MAP_NAME):
    """
    Visualize the clusters in a timeserie (2D: time x range).

    Args:
        dataset (xarray dataset): _description_
        original_shape_labels_array (_type_): _description_
        num_categories (_type_): _description_
        fig (_type_, optional): _description_. Defaults to None.
        ax (_type_, optional): _description_. Defaults to None.
        range_limits (list, optional): _description_. Defaults to [0, 5000].
        color_map (_type_, optional): _description_. Defaults to COLOR_MAP_NAME.

    Returns:
        _type_: _description_
    """
    if (ax is None) | (fig is None):
        set_fig = True
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))  # Set the figure size as needed
    else:
        set_fig = False

    # Create a colored mesh plot using the custom colormap
    bounds = np.arange(0, num_categories + 1)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    plot = ax.pcolormesh(dataset['time'], dataset['range'], original_shape_labels_array, norm=norm, shading='nearest',
                         cmap=color_map)

    # Add a colorbar with discrete color labels
    # norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cbar = fig.colorbar(plot, ax=ax, cmap=color_map, ticks=range(num_categories))
    cbar.set_ticklabels(range(num_categories))  # Set labels to category indices

    # Set labels for x and y axes (if needed)
    ax.set_ylim(range_limits)
    ax.set_xlabel('Time')
    ax.set_ylabel('Range [m]')

    # Set the title of the plot (if needed)
    # ax.set_title('K-Means Clustering Results')

    # Rotate time labels (xticks)
    ax.tick_params(axis='x', rotation=45, which='major')

    # Show the plot
    #if set_fig:
    #    print("showing fig")
    #    plt.show()

    return ax, fig


def plot_classified_timeserie(classified_array,
                              time_array,
                              range_array,
                              config,
                              ylims=[0, 10000],
                              fig=None, ax=None,
                              save_fig=False):
    """
    Plot the classified timeseries using a colored mesh plot with class IDs and names.

    Args:
        classified_array (ndarray): Array containing classified data.
        time_array (ndarray): Array containing time information.
        range_array (ndarray): Array containing range information.
        config (dict): Configuration dictionary containing class information.
        ylims (list, optional): List specifying the range limits. Defaults to [0, 10000].
        fig (matplotlib.figure.Figure, optional): Matplotlib figure object. Defaults to None.
        ax (matplotlib.axes._axes.Axes, optional): Matplotlib axes object. Defaults to None.
        save_fig (bool, optional): Flag to save the figure. Defaults to False.

    Returns:
        matplotlib.figure.Figure, matplotlib.axes._axes.Axes: Figure and axes objects.
    """

    # Get necessary values from config dictionnary
    category_colors = [category['color'] for category in config['classes']]
    category_ids = [category['class_id'] for category in config['classes']]

    # Set each category color into a colormap
    classification_cmap = ListedColormap(category_colors)
    num_categories = len(config['classes'])

    # Create figure if necessary
    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        set_figure = True
    else :
        set_figure = False
        
    # Create a colored mesh plot using the custom colormap
    plot = ax.pcolormesh(time_array, range_array, classified_array,
                         cmap=classification_cmap, vmin=0, vmax=num_categories, shading='nearest')

    # Add a colorbar with discrete color labels
    cbar = fig.colorbar(plot, cmap=classification_cmap)
    cbar.set_ticks([i + 0.5 for i in category_ids])
    cbar.set_ticklabels([f"{category['class_id']}: {category['class_name']}" for category in config['classes']])

    # Set labels for x and y axes
    ax.set_ylim(ylims)
    ax.set_xlabel('Time')
    ax.set_ylabel('Range [m]')

    # Set the title of the plot
    # ax.set_title('Classification Results')

    if save_fig:
        filepath = filename_to_save if isinstance(filename_to_save, str) else f"classified_timeserie_{time_array[0]}.jpg"
        print(f'Saving element to {filepath}')
        plt.savefig(filepath, bbox_inches='tight', dpi=300)

    # Show the plot
    #if set_figure:
    #    plt.show()

    return fig, ax
