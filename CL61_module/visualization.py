#array management
import numpy as np
import xarray as xr
import pandas as pd

#Basic plots
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap

# batlow colourmap
import cmcrameri.cm as cmc 

#improved large data visualization
import datashader as ds
from datashader import transfer_functions as tf, reductions as rd


COLOR_MAP_DEFAUT = 'cmc.batlow'

BATLOW_7COLORS  = [{"name":"Fairy Tale","hex":"FBC5E6","rgb":[251,197,230],"cmyk":[0,22,8,2],"hsb":[323,22,98],"hsl":[323,87,88],"lab":[85,24,-9]},
                   {"name":"Olive","hex":"88842B","rgb":[136,132,43],"cmyk":[0,3,68,47],"hsb":[57,68,53],"hsl":[57,52,35],"lab":[54,-10,47]},
                   {"name":"Fern green","hex":"4C734B","rgb":[76,115,75],"cmyk":[34,0,35,55],"hsb":[119,35,45],"hsl":[119,21,37],"lab":[45,-22,18]},
                   {"name":"Penn Blue","hex":"03245C","rgb":[3,36,92],"cmyk":[97,61,0,64],"hsb":[218,97,36],"hsl":[218,94,19],"lab":[16,14,-37]},
                   {"name":"Butterscotch","hex":"D69444","rgb":[214,148,68],"cmyk":[0,31,68,16],"hsb":[33,68,84],"hsl":[33,64,55],"lab":[66,17,51]},
                   {"name":"Melon","hex":"FCAC99","rgb":[252,172,153],"cmyk":[0,32,39,1],"hsb":[12,39,99],"hsl":[12,94,79],"lab":[78,27,22]},
                   {"name":"Midnight green","hex":"115362","rgb":[17,83,98],"cmyk":[83,15,0,62],"hsb":[191,83,38],"hsl":[191,70,23],"lab":[32,-14,-14]}]

COLOR_CODES_BLUE_YEL = ['#03245C', '#D69444']

def plotCL61AsColomersh(dataset, variable_names=['beta_att', 'linear_depol_ratio'],
                        time_var = 'time',
                        range_var = 'range',
                        min_value=1e-7,
                        max_value=1e-4,
                        hlims=False,
                        color_map = 'Spectral_r',
                        scales = ['log', 'linear'],):
    
    x = dataset[time_var].values
    h = np.round(dataset[range_var].values)
    back_att_arr = dataset[variable_names[0]].T.values
    depol_arr = dataset[variable_names[1]].T.values

    fig, [ax, ax2] = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    
    lims = [np.max([np.nanmin(back_att_arr), min_value]), np.min([np.nanmax(back_att_arr), max_value])]

    if hlims:
        if len(hlims)<2:
            hlims = [0,hlims]
        elif len(hlims)>2:
            print('Did not expect hlims length > 2 ; taking first 2 values')
            hlims = hlims[:2]
    else :
        hlims = [0,15000]
    if scales[0]=='log':
        cax = ax.pcolormesh(x, h, back_att_arr, axes = ax, shading='nearest', cmap=color_map,
                            norm=colors.LogNorm(vmin=lims[0], vmax=lims[1]))
    else:
        cax = ax.pcolormesh(x, h, back_att_arr, axes = ax, shading='nearest', cmap=color_map, vmin =0, vmax = 1)  
    cbar = fig.colorbar(cax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(rf'{variable_names[0]}', rotation=90)
    ax.set_ylabel('Range (m)')
    #ax.set_xlabel('time')
    ax.set_ylim(hlims)
    
    if scales[1]=='log':
        cax2 = ax2.pcolormesh(x, h, depol_arr, shading='nearest', cmap=color_map,
                            norm=colors.LogNorm(vmin=lims[0], vmax=lims[1]))
    else:
        cax2 = ax2.pcolormesh(x, h, depol_arr, shading='nearest', cmap=color_map, vmin=0, vmax=1)  
        
    cbar = fig.colorbar(cax2)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(rf'{variable_names[1]}', rotation=90)
    ax2.set_ylabel('Range (m)')
    ax2.set_xlabel('time')
    ax2.set_ylim(hlims)


    plt.show()
    return

def plotCl61asProfile(dataset, time_period = None, variable='beta_att', hlims = False, color_map = 'Spectral_r'):

    if type(variable) == str:
        L = 1
        variable = [variable]
    elif type(variable) == list:
        L = len(variable)
    else :
        print(f"Didn't expect variable of type {type(variable)}")
        return False

    if time_period:
        if type(time_period) == str:
            subset = dataset.sel(time = time_period, method="nearest")
        elif type(time_period) == list:
            if len(time_period)>2:
                print(f"Expected max 2 values (start time/date, end time/date) but got {len(time_period)}")
                return 0
            else:
                subset = dataset.sel(time=slice(time_period[0], time_period[1]))
    else:
        subset = dataset

    
    fig, ax = plt.subplots(1,L, sharey = True)
    if L==1:
        ax = [ax]

    h = np.round(dataset['range'].values)

    for var_i in range(L):
        
        x = subset[variable[var_i]].T.values
        ax[var_i].plot(x, h, alpha=0.5)

        # should be 3276,T
        if x.ndim > 2:
            print('Did not expect dim>2')
        elif x.ndim == 2:
            mean_x = np.mean(x, axis=1)
            ax[var_i].plot(mean_x, h, color='red')

    
        ax[var_i].set_xlabel(f'{variable}')
        ax[var_i].set_ylabel('range')
        ax[var_i].set_title(f'{variable} profile')

    if hlims:
        if type(len)==int:
            hlims = [0,hlims]
        elif len(hlims)>2:
            print('Did not expect hlims length > 2 ; taking first 2 values')
            hlims = hlims[:2]
        elif type(hlims)!=list: 
            print("Input hlims type not matching")
            return False
        plt.ylim(hlims)


    return True

def plotVerticalProfiles(dataset, time_period = None,
                          var_names = ['beta_att', 'linear_depol_ratio', 'range'],
                          hlims = [0, 15000],
                          xlabel1='Beta attenuation',
                          xlabel2 = 'linear depol ratio',
                          ylabel = 'range [m]',
                          title = 'CL61 profiles',
                          var_xlims = [[1e-7, 1e-4],[0,1]],
                          x_scales = ['log', 'linear'],
                          plot_colors = ['#124E63', '#F6A895'],
                          ax = None):
    '''
    Plot profiles of beta attenuation and depolarization ratio respect to height (range).
    '''
    # Get time period wished
    if time_period:
        if isinstance(time_period, str):
            subset = dataset.sel(time=time_period, method="nearest")
        elif isinstance(time_period, list):
            if len(time_period) > 2:
                raise ValueError(f"Expected max 2 values (start time/date, end time/date) but got {len(time_period)}")
            subset = dataset.sel(time=slice(time_period[0], time_period[1])).mean(dim='time')
        else:
            raise TypeError(f"Expected time_period to be of type str or list, but got {type(time_period)}")
    else:
        subset = dataset.mean(dim='time')


    # Get variables
    beta_atts = subset[var_names[0]]
    lin_depols = subset[var_names[1]]
    heights = subset[var_names[2]]

    # Create a figure and two subplots
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4,6))

    # Plot the first variable with a logarithmic x-axis
    if x_scales[0] == 'log':
        ax.semilogx(beta_atts, heights, label=xlabel1, color= plot_colors[0], alpha = 0.8)
    else:
        ax.plot(beta_atts, heights, label=xlabel1, color=plot_colors[0], alpha = 0.8)
    ax.set_xlabel(xlabel1)
    ax.set_ylabel(ylabel)
    ax.set_xlim(var_xlims[0])

    # Plot the second variable with a linear x-axis on a separate axis
    ax2 = ax.twiny()
    ax2.tick_params(axis='x', labelcolor=plot_colors[1])
    if x_scales[1] == 'log':
        ax2.semilogx(lin_depols, heights, label=xlabel2, color= plot_colors[1], alpha = 0.8)
    else:
        ax2.plot(lin_depols, heights, label=xlabel2, color=plot_colors[1], alpha = 0.8)
    ax2.set_xlabel(xlabel2, color = plot_colors[1])
    ax2.set_xlim(var_xlims[1])


    # Add a legend
    #ax1.legend(loc='upper right')
    #ax2.legend(loc='upper right')

    # set_title:
    plt.title(title)

    # set height limits
    plt.ylim(hlims)

    # Show the plot
    if ax == None:
        plt.show()

    return [ax, ax2]


def plot_classifiction_result(dataset, classified_var_name = 'classified',
                              colormap = COLOR_MAP_DEFAUT):

    # Define the original colormap (e.g., 'viridis')
    original_cmap = plt.get_cmap(colormap)

    # Define the number of discrete categories
    num_categories = dataset['classified'].unique().size  # Adjust as needed

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
    plt.title('K-Means Clustering Results')

    # Show the plot
    plt.show()
    
    return plot
    
def plot_2D_classification():
    # Assuming 'cleaned_feature_matrix' is your feature matrix
    df = pd.DataFrame({'log10_Beta_att': cleaned_feature_matrix[:, 0],
                    'Linear_depol': cleaned_feature_matrix[:, 1],
                    'Cluster_Labels': cluster_labels})

    cvs = ds.Canvas(plot_width=500, plot_height=500)
    agg = cvs.points(df, 'log10_Beta_att', 'Linear_depol', ds.mean('Cluster_Labels'))

    # Use the color map you defined earlier
    colors_rgb = (discrete_colors[:,0:3] * 255).astype(int)
    color_key = dict(enumerate(colors_rgb))

    # Create a custom colormap in a format that Datashader can use
    #cmap = tf.color_map(builders=[colors_rgb.tolist()], name="custom_cmap")


    img = tf.shade(agg, cmap=cmc.batlow)

    # Display the plot
    tf.set_background(img, 'black')