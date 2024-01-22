#array management
import numpy as np
import pandas as pd

#Basic plots
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap

# batlow colourmap
import cmcrameri.cm as cmc 

#improved large data visualization
import datashader as ds
from datashader.mpl_ext import dsshow

# Needed function
from scipy.stats import mode # for hexbins better plotting 
from .utils import load_config, filename_to_save

# Needed global variable
config_classification = None 

COLOR_MAP_NAME = 'cmc.batlow'
COLOR_MAP = cmc.batlow # type: ignore
BATLOW_7COLORS  = [{"name":"Fairy Tale","hex":"FBC5E6","rgb":[251,197,230],"cmyk":[0,22,8,2],"hsb":[323,22,98],"hsl":[323,87,88],"lab":[85,24,-9]},
                   {"name":"Olive","hex":"88842B","rgb":[136,132,43],"cmyk":[0,3,68,47],"hsb":[57,68,53],"hsl":[57,52,35],"lab":[54,-10,47]},
                   {"name":"Fern green","hex":"4C734B","rgb":[76,115,75],"cmyk":[34,0,35,55],"hsb":[119,35,45],"hsl":[119,21,37],"lab":[45,-22,18]},
                   {"name":"Penn Blue","hex":"03245C","rgb":[3,36,92],"cmyk":[97,61,0,64],"hsb":[218,97,36],"hsl":[218,94,19],"lab":[16,14,-37]},
                   {"name":"Butterscotch","hex":"D69444","rgb":[214,148,68],"cmyk":[0,31,68,16],"hsb":[33,68,84],"hsl":[33,64,55],"lab":[66,17,51]},
                   {"name":"Melon","hex":"FCAC99","rgb":[252,172,153],"cmyk":[0,32,39,1],"hsb":[12,39,99],"hsl":[12,94,79],"lab":[78,27,22]},
                   {"name":"Midnight green","hex":"115362","rgb":[17,83,98],"cmyk":[83,15,0,62],"hsb":[191,83,38],"hsl":[191,70,23],"lab":[32,-14,-14]}]
COLOR_CODES_BLUE_YEL = ['#03245C', '#D69444']



def visualize_classification_featurespace_2D(log10_beta_att_flatten,
                                             linear_depol_ratio_flatten,
                                             cluster_labels_flatten,
                                             fig = None, ax = None,
                                             plot_cbar = True,
                                             use_datashader = False,
                                             cmap = COLOR_MAP):

    # Assuming 'cleaned_feature_matrix' is your feature matrix
    df = pd.DataFrame({'log10_beta': log10_beta_att_flatten,
                    'LDR': linear_depol_ratio_flatten,
                    'cluster_labels': cluster_labels_flatten})

    #cvs = ds.Canvas(plot_width=500, plot_height=500)
    if (ax==None) | (fig==None):
        print("new figure") #debugging
        set_figure = True
        fig, ax = plt.subplots(1,1, figsize=(10, 10))
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
    
    if use_datashader:
        print("Using datashader for plotting ") #debugging 
            # Get range of values to reset axes labels
        x_range = (np.nanmin(log10_beta_att_flatten), np.nanmax(log10_beta_att_flatten))
        y_range = (np.nanmin(linear_depol_ratio_flatten), np.nanmax(linear_depol_ratio_flatten))
    
        artist = dsshow(df, ds.Point('log10_beta_att', 'LDR'), ds.mean('cluster_labels'),
                        ax=ax, cmap = discrete_cmap)
    else:
        def most_freq_value(a):
            return np.bincount(a).argmax()
        
        artist = ax.hexbin(log10_beta_att_flatten, linear_depol_ratio_flatten, C=cluster_labels_flatten,
                           gridsize=500, cmap=discrete_cmap,
                           reduce_C_function=most_freq_value, mincnt=2)
    
    if plot_cbar:
        fig.colorbar(artist, ax=ax, orientation='vertical')
        
    #ax.set_title('Clustering in Feature space')
    ax.set_xlabel('$Log_{10}$ attenuated backscatter \n coefficient [$m^{-1}~sr^{-1}$]')
    ax.set_ylabel('Linear depolarization ratio')

    
    if set_figure:
        plt.show()

    return fig, ax

def visualize_cluster_results(dataset,
                            original_shape_labels_array,
                            num_categories,
                            fig = None, 
                            ax = None,
                            range_limits = [0,5000],
                            color_map = COLOR_MAP_NAME):
    

    if (ax==None) | (fig==None): 
        set_fig = True
        fig, ax = plt.subplots(1,1, figsize=(8, 5))  # Set the figure size as needed
    else:
        set_fig = False

    # Create a colored mesh plot using the custom colormap
    bounds = np.arange(0,num_categories+1)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    plot = ax.pcolormesh(dataset['time'], dataset['range'], original_shape_labels_array, norm = norm, shading='nearest', cmap=color_map)

    # Add a colorbar with discrete color labels
    #norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cbar = fig.colorbar(plot, ax=ax, cmap=color_map, ticks=range(num_categories))
    cbar.set_ticklabels(range(num_categories))  # Set labels to category indices

    # Set labels for x and y axes (if needed)
    ax.set_ylim(range_limits)
    ax.set_xlabel('Time')
    ax.set_ylabel('Range [m]')
    
    # Set the title of the plot (if needed)
    #ax.set_title('K-Means Clustering Results')

    # Rotate time labels (xticks)
    ax.tick_params(axis='x', rotation=45, which='major')
    
    # Show the plot
    if set_fig:
        print("showing fig")
        plt.show()

    return ax, fig

def colormesh_classification_results(dataset, classified_var_name = 'classified', title = None, 
                              colormap = COLOR_MAP_NAME):

    # Define the original colormap (e.g., 'viridis')
    original_cmap = plt.get_cmap(colormap)

    # Define the number of discrete categories
    num_categories = len(np.unique(dataset[classified_var_name]))  # Adjust as needed

    # Create a list of evenly spaced values to sample the colormap
    color_values = np.linspace(0, 1, num_categories)

    # Sample the original colormap at the specified values
    discrete_colors = original_cmap(color_values)

    # Create a custom ListedColormap with the discrete colors
    discrete_cmap = ListedColormap(discrete_colors)
    
    plt.figure(figsize=(8, 5))  # Set the figure size as needed

    # Create a colored mesh plot using the custom colormap
    plot = plt.pcolormesh(dataset['time'], dataset['range'], dataset[classified_var_name].T,
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
    if title:
        plt.title(title)

    # Show the plot
    plt.show()
    
    return plot


def plot_classified_colormesh_old(classified_array,
                              config_dictionary,
                              time_array,
                              range_array,
                              ylims = [0,5000],
                              fig = None, ax = None,
                              title = None):
    
    # Get necessary values from config file/variable
    config_classification = load_config()
    category_colors = config_classification['category_colors']
    classification_cmap = ListedColormap(category_colors)

    class_combination_mapping = config_classification['class_combination_mapping']
    num_categories = len(class_combination_mapping)

    if (fig==None)|(ax==None):
        fig, ax = plt.subplots(1,1, figsize=(10, 5))  # Set the figure size as needed

    # Create a colored mesh plot using the custom colormap
    plot = ax.pcolormesh(time_array, range_array, classified_array,
                cmap=classification_cmap, vmin=0, vmax=num_categories,shading='nearest')

    # Add a colorbar with discrete color labels
    #norm = plt.Normalize(vmin=0, vmax=num_categories)
    cbar = fig.colorbar(plot, cmap=classification_cmap)
    cbar.set_ticks(np.linspace(0, num_categories-1, num_categories)+0.5)
    cbar.set_ticklabels(list(class_combination_mapping.values()))  # Set labels to category indices

    # Set labels for x and y axes (if needed)
    ax.set_ylim(ylims)
    ax.set_xlabel('Range')
    ax.set_ylabel('Time')
    
    # Set the title of the plot (if needed)
    if title:
        ax.set_title(title)
    
    # Rotate time labels (xticks)
    ax.tick_params(axis='x', rotation=45, which='major')

    # Show the plot
    if (fig==None)|(ax==None):
        plt.show()
    return fig, ax

    
def plot_classified_timeserie(classified_array, time_array, range_array, config, ylims=[0, 10000], fig=None, ax=None, save_fig = False):
    '''Plots the classifed array with id corresponding to class naem as given in config file'''
    # Get necessary values from the config file
    category_colors = [category['color'] for category in config['classes']]
    category_ids = [category['class_id'] for category in config['classes']]
    classification_cmap = ListedColormap(category_colors)

    num_categories = len(config['classes'])

    if (fig is None) or (ax is None):
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))  # Set the figure size as needed

    # Create a colored mesh plot using the custom colormap
    plot = ax.pcolormesh(time_array, range_array, classified_array,
                         cmap=classification_cmap, vmin=0, vmax=num_categories, shading='nearest')

    # Add a colorbar with discrete color labels
    cbar = fig.colorbar(plot, cmap=classification_cmap)
    cbar.set_ticks([i+0.5 for i in category_ids])
    cbar.set_ticklabels([f"{category['class_id']}: {category['class_name']}" for category in config['classes']])

    # Set labels for x and y axes (if needed)
    ax.set_ylim(ylims)
    ax.set_xlabel('Time')
    ax.set_ylabel('Range [m]')

    # Set the title of the plot (if needed)
    #ax.set_title('Classification Results')

    # Show the plot
    if (fig is None) or (ax is None):
        plt.show()
    
    return fig, ax