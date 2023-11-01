# array management
import numpy as np
import pandas as pd

# Basic plots
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import seaborn as sns

# batlow colourmap
import cmcrameri.cm as cmc

# improved large data visualization
import datashader as ds
from datashader.mpl_ext import dsshow  # for matplotlib integration
from datashader import transfer_functions as tf

from .classification_vizalization import *
from .utils import filename_to_save

COLOR_MAP_NAME = 'cmc.batlow'
COLOR_MAP = cmc.batlow  # type: ignore

BATLOW_7COLORS = [
    {"name": "Fairy Tale", "hex": "FBC5E6", "rgb": [251, 197, 230], "cmyk": [0, 22, 8, 2], "hsb": [323, 22, 98],
     "hsl": [323, 87, 88], "lab": [85, 24, -9]},
    {"name": "Olive", "hex": "88842B", "rgb": [136, 132, 43], "cmyk": [0, 3, 68, 47], "hsb": [57, 68, 53],
     "hsl": [57, 52, 35], "lab": [54, -10, 47]},
    {"name": "Fern green", "hex": "4C734B", "rgb": [76, 115, 75], "cmyk": [34, 0, 35, 55], "hsb": [119, 35, 45],
     "hsl": [119, 21, 37], "lab": [45, -22, 18]},
    {"name": "Penn Blue", "hex": "03245C", "rgb": [3, 36, 92], "cmyk": [97, 61, 0, 64], "hsb": [218, 97, 36],
     "hsl": [218, 94, 19], "lab": [16, 14, -37]},
    {"name": "Butterscotch", "hex": "D69444", "rgb": [214, 148, 68], "cmyk": [0, 31, 68, 16], "hsb": [33, 68, 84],
     "hsl": [33, 64, 55], "lab": [66, 17, 51]},
    {"name": "Melon", "hex": "FCAC99", "rgb": [252, 172, 153], "cmyk": [0, 32, 39, 1], "hsb": [12, 39, 99],
     "hsl": [12, 94, 79], "lab": [78, 27, 22]},
    {"name": "Midnight green", "hex": "115362", "rgb": [17, 83, 98], "cmyk": [83, 15, 0, 62], "hsb": [191, 83, 38],
     "hsl": [191, 70, 23], "lab": [32, -14, -14]}]

COLOR_CODES_BLUE_YEL = ['#03245C', '#D69444']


def histogram1d(dataset,
                variable_name='beta_att_clean',
                hue_variable=None,
                var_transform='log',
                count_log=False,
                use_matplotlib=False,
                save_fig=True,
                cmap=COLOR_MAP,
                **kwargs):
    """
    Create a 1D histogram or bar plot for a given variable.

    Parameters
    ----------
    dataset : xarray dataset

    variable_name : str
        default = beta_att_clean
    hue_variable : str
         The name of the variable to use for color categorization.
    var_transform: 'str'
        'log' for logarithmic scale on x-axis, None otherwise
    count_log: boolean
        logarithmic count scale
    use_matplotlib: boolean
        If True, use Matplotlib for plotting. Otherwise, use Seaborn.
    save_fig: boolean
        If fig should be saved or not
    cmap : matplotlib colormap
    **kwargs :
        Additional keyword arguments to pass to the plotting function.

    Returns
    -------
        None
    """

    # Manage scales:
    if var_transform == 'log':
        if count_log:
            log_scales = [True, True]
        else:
            log_scales = True
    else:
        log_scales = False

    # Generate some example data
    x = dataset[variable_name].values.flatten()

    if hue_variable:
        c = dataset[hue_variable].values.flatten()
        df = pd.DataFrame({variable_name: x, hue_variable: c})

    else:
        c = None
        df = pd.DataFrame({variable_name: x})

    if use_matplotlib:
        # Use Matplotlib for more fine-grained control
        plt.figure(figsize=(8, 6))
        plt.title(f'1D Histogram of {variable_name}')
        plt.xlabel(variable_name)
        plt.ylabel('Count (log scale)' if count_log else 'Count')

        if var_transform == 'log':
            df[variable_name] = np.log10(df[variable_name])

        if hue_variable:
            for hue_value in df[hue_variable].unique():
                data = df[df[hue_variable] == hue_value][variable_name]
                plt.hist(data, alpha=0.5, label=f'{hue_variable}={hue_value}', log=count_log, **kwargs)
            plt.legend()
        else:
            plt.hist(df[variable_name], log=count_log, **kwargs)

    else:
        # Use Seaborn for a simple, attractive plot
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))

        sns.histplot(data=df, x=variable_name, hue=hue_variable, multiple="stack",
                     palette=(cmap if hue_variable else None), edgecolor=".3",
                     linewidth=.5, log_scale=log_scales, **kwargs)
        plt.title(f'1D Histogram of {variable_name}')
        plt.xlabel(variable_name)
        plt.ylabel('Count (log scale)' if count_log else 'Count')

    if save_fig:
        plt.savefig(filename_to_save(dataset, save_fig, suffix=f'hist_{variable_name}'))

    plt.show()
    return


def histogram2d(dataset,
                var1 = 'beta_att_clean',
                var2 = 'linear_depol_ratio_clean',
                log_transforms=[True, False],
                count_tf='log',
                min_count_per_bin=2,
                cmap=COLOR_MAP,
                save_fig = True):
    """
    Plots the 2D histogram for given variables

    Parameters
    ----------
    dataset
    variable_names
    log_transforms
    count_tf
    min_count_per_bin
    cmap

    Returns
    -------
        seaborn JointGridObject
    """
    if (var1 is None)|(var2 is None):
        raise (ValueError('This function is designed for 2 dimensions'))

    # Generate some example data
    x = dataset[var1].values.flatten()
    # Apply transforms
    if log_transforms[0]:
        x = np.log10(x)
    y = dataset[var2].values.flatten()
    if log_transforms[1]:
        y = np.log10(y)
    # Into a dataframe for seaborn use
    df = pd.DataFrame({var1: x, var2: y})

    # Create a jointGrid
    g = sns.JointGrid(data=df, x=var1, y=var2, height=8, ratio=4)
    # Plot hist as hexbin
    hb = g.ax_joint.hexbin(x, y, gridsize=500, cmap=cmap,
                           mincnt=min_count_per_bin, bins=count_tf)

    # Add a colorbar to show the mapping of z values to colors
    cax = plt.gcf().add_axes([1.05, 0.1, 0.05, 0.7])  # Adjust the position and size of the colorbar
    cbar = plt.colorbar(hb, cax=cax)

    # Create marginal histograms
    sns.histplot(data=df, x=var1, ax=g.ax_marg_x, color='#03245C', cbar_kws={"edgecolor": None})
    sns.histplot(data=df, y=var2, ax=g.ax_marg_y, color='#03245C', orientation='horizontal', cbar_kws={"edgecolor": None})

    if save_fig:
        filepath = filename_to_save(dataset, save_fig, suffix='hist2D')
        print(f'saved element to {filepath}')
        plt.savefig(filepath, bbox_inches='tight')

    return g


def plotCL61AsColomersh(dataset, variable_names=['beta_att', 'linear_depol_ratio'],
                        time_var='time',
                        range_var='range',
                        min_value=1e-7,
                        max_value=1e-4,
                        range_limits=False,
                        color_map=COLOR_MAP_NAME,
                        scales=['log', 'linear']):
    """

    Parameters
    ----------
    dataset
    variable_names
    time_var
    range_var
    min_value
    max_value
    range_limits
    color_map
    scales

    Returns
    -------

    """
    x = dataset[time_var].values
    h = np.round(dataset[range_var].values)
    back_att_arr = dataset[variable_names[0]].T.values
    depol_arr = dataset[variable_names[1]].T.values

    fig, [ax, ax2] = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    lims = [np.max([np.nanmin(back_att_arr), min_value]), np.min([np.nanmax(back_att_arr), max_value])]

    if range_limits:
        if len(range_limits) < 2:
            range_limits = [0, range_limits]
        elif len(range_limits) > 2:
            print('Did not expect range_limits length > 2 ; taking first 2 values')
            range_limits = range_limits[:2]
    else:
        range_limits = [0, 15000]
    if scales[0] == 'log':
        cax = ax.pcolormesh(x, h, back_att_arr, axes=ax, shading='nearest', cmap=color_map,
                            norm=colors.LogNorm(vmin=lims[0], vmax=lims[1]))
    else:
        cax = ax.pcolormesh(x, h, back_att_arr, axes=ax, shading='nearest', cmap=color_map, vmin=0, vmax=1)
    cbar = fig.colorbar(cax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(rf'{variable_names[0]}', rotation=90)
    ax.set_ylabel('Range (m)')
    # ax.set_xlabel('time')
    ax.set_ylim(range_limits)

    if scales[1] == 'log':
        cax2 = ax2.pcolormesh(x, h, depol_arr, shading='nearest', cmap=color_map,
                              norm=colors.LogNorm(vmin=lims[0], vmax=lims[1]))
    else:
        cax2 = ax2.pcolormesh(x, h, depol_arr, shading='nearest', cmap=color_map, vmin=0, vmax=1)

    cbar = fig.colorbar(cax2)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(rf'{variable_names[1]}', rotation=90)
    ax2.set_ylabel('Range (m)')
    ax2.set_xlabel('time')
    ax2.set_ylim(range_limits)

    plt.show()
    return


def plotCL61AsColomersh_ds(dataset, variable_names=['beta_att', 'linear_depol_ratio'],
                           time_var='time',
                           range_var='range',
                           min_value=1e-7,
                           max_value=1e-4,
                           range_limits=False,
                           color_map=COLOR_MAP_NAME,
                           scales=['log', 'linear'],
                           ax = None,
                           fig = None):
    '''
    Test to plot colormesh with datashader for more efficient visualization
    '''

    # cvs = ds.Canvas(plot_width=500, plot_height=500)
    if (ax == None) | (fig == None):
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Define the original colormap (e.g., 'viridis')
    original_cmap = plt.get_cmap(color_map)

    # Define the number of discrete categories
    num_categories = np.unique(dataset['cluster_labels']).size  # Adjust as needed

    # Create a list of evenly spaced values to sample the colormap
    color_values = np.linspace(0, 1, num_categories)

    # Sample the original colormap at the specified values
    discrete_colors = original_cmap(color_values)

    # Create a custom ListedColormap with the discrete colors
    discrete_cmap = ListedColormap(discrete_colors)

    artist0 = dsshow(dataset, tf.shade('log10_beta_att', 'linear_depol'), ds.mean('cluster_labels'),
                     ax=ax, cmap=discrete_cmap)

    # agg = cvs.points(df, 'log10_Beta_att', 'Linear_depol', ds.mean('Cluster_Labels'))
    if (ax == None) | (fig == None):
        fig.colorbar(artist0, ax=ax, orientation='vertical')
    ax.set_title('Feature space clustering')
    ax.set_xlabel('log10 beta attenuation')
    ax.set_ylabel('linear depolarisation')

    return


def plotCl61asProfile(dataset, time_period=None,
                      variable='beta_att',
                      range_limits=False,
                      color_map='Spectral_r'):
    if type(variable) == str:
        L = 1
        variable = [variable]
    elif type(variable) == list:
        L = len(variable)
    else:
        print(f"Didn't expect variable of type {type(variable)}")
        return False

    if time_period:
        if type(time_period) == str:
            subset = dataset.sel(time=time_period, method="nearest")
        elif type(time_period) == list:
            if len(time_period) > 2:
                print(f"Expected max 2 values (start time/date, end time/date) but got {len(time_period)}")
                return 0
            else:
                subset = dataset.sel(time=slice(time_period[0], time_period[1]))
    else:
        subset = dataset

    fig, ax = plt.subplots(1, L, sharey=True)
    if L == 1:
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

    if range_limits:
        if type(len) == int:
            range_limits = [0, range_limits]
        elif len(range_limits) > 2:
            print('Did not expect range_limits length > 2 ; taking first 2 values')
            range_limits = range_limits[:2]
        elif type(range_limits) != list:
            print("Input range_limits type not matching")
            return False
        plt.ylim(range_limits)

    return True


def plotVerticalProfiles(dataset, time_period=None,
                         var_names=['beta_att', 'linear_depol_ratio'],
                         range_limits=[0, 15000],
                         xlabel1='Beta attenuation',
                         xlabel2='linear depol ratio',
                         ylabel='range [m]',
                         title='CL61 profiles',
                         var_xlims=[[1e-7, 1e-4],
                                    [0, 1]],
                         x_scales=['log', 'linear'],
                         plot_colors=['#124E63', '#F6A895'],
                         ax=None):
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
    heights = subset['range']

    # Create a figure and two subplots
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 6))

    # Plot the first variable with a logarithmic x-axis
    if x_scales[0] == 'log':
        ax.semilogx(beta_atts, heights, label=xlabel1, color=plot_colors[0], alpha=0.8)
    else:
        ax.plot(beta_atts, heights, label=xlabel1, color=plot_colors[0], alpha=0.8)

    ax.set_xlabel(xlabel1)
    ax.set_ylabel(ylabel)
    ax.set_xlim(var_xlims[0])

    # Plot the second variable with a linear x-axis on a separate axis
    ax2 = ax.twiny()
    ax2.tick_params(axis='x', labelcolor=plot_colors[1])
    if x_scales[1] == 'log':
        ax2.semilogx(lin_depols, heights, label=xlabel2, color=plot_colors[1], alpha=0.8)
    else:
        ax2.plot(lin_depols, heights, label=xlabel2, color=plot_colors[1], alpha=0.8)
    ax2.set_xlabel(xlabel2, color=plot_colors[1])
    ax2.set_xlim(var_xlims[1])

    # Add a legend
    # ax1.legend(loc='upper right')
    # ax2.legend(loc='upper right')
    # set_title:
    plt.title(title)

    # set height limits
    plt.ylim(range_limits)

    # Show the plot
    if ax == None:
        plt.show()

    return [ax, ax2]
