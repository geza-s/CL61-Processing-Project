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

class PlotCL61:
    
    def __init__(self, dataset):
        self.dataset = dataset

    def show_timeserie(self,
                        variable_names=['beta_att', 'linear_depol_ratio'],
                        label_names = None,
                        value_ranges=[[1e-7, 1e-4],[0,1]],
                        range_limits=None,
                        scales=['log', 'linear'],
                        color_map="cmc.batlow",
                        fig = None, axs = None,
                        save_fig = False,
                        fig_dpi = 300,
                        **kwargs):
        
        # Slice data based on range
        range_limits = get_range_limits(range_limits)
        subset = self.dataset.sel(range = slice(*range_limits))

        # Check variables to plot
        if isinstance(variable_names,str):
            variable_names = [variable_names]
        L = len(variable_names)

        # Plot 
        if axs is None:
            fig, axs = plt.subplots(L, 1, sharex=True, figsize = (12, L*3 + 1))
        elif fig is None:
            fig = plt.figure(figsize = (12, L*3 + 1))

        for i, var_name in enumerate(variable_names):

            # Get ax
            try:
                ax = axs[i]
            except (TypeError, IndexError):
                ax = fig.add_subplot(L, 1, i+1)

            # Get range of value
            try: vmin, vmax = value_ranges[i]
            except: vmin, vmax = None, None

            if scales[i] == 'log':
                vmin = np.log10(vmin) if vmin != None  else vmin
                vmax = np.log10(vmax) if vmax != None  else vmax
                im = np.log10(subset[var_name]).plot.imshow(x='time', y='range', vmin=vmin, vmax=vmax, ax = ax, cmap=color_map)
            else:
                im = subset[var_name].plot.imshow(x='time', y='range', vmin=vmin, vmax=vmax, ax = ax, cmap = color_map)

            cbar = im.colorbar
            if label_names:
                cbar.set_label(var_name[i])
            #else:
            #    cbar.set_label(var_name)
            ax.set_ylabel('range [m]')
            ax.set_title('')

        if save_fig:
            filepath = filename_to_save(self.dataset, save_fig, suffix='colormesh')
            print(f'saved figure to {filepath}')
            plt.savefig(filepath, bbox_inches='tight', dpi=fig_dpi)

        return axs


    def colormesh(self, variable_names=['beta_att', 'linear_depol_ratio'],
                        min_value=1e-7,
                        max_value=1e-4,
                        range_limits=None,
                        scales=['log', 'linear'],
                        time_var='time',
                        range_var='range',
                        color_map=COLOR_MAP_NAME,
                        save_fig = False,
                        fig_dpi = 300,
                        fig = None, axs = None,
                        **kwargs):
        """
        Plot CL61 data as a colormesh.

        Args:
            dataset (_type_): The dataset to be plotted.
            variable_names (list, optional): Names of the variables to be plotted.. Defaults to ['beta_att', 'linear_depol_ratio'].
            min_value (_type_, optional):Minimum value for the color scale.. Defaults to 1e-7.
            max_value (_type_, optional): Maximum value for the color scale.. Defaults to 1e-4.
            range_limits (int or list , optional): max_range or [min_range, max_range] for the y-axis (optional upper limit).. Defaults to None.
            color_map (_type_, optional): Matplotlib colormap name. (Default see file).
            scales (list, optional): to specify the color scales. Defaults to ['log', 'linear'].
            time_var (str, optional): Name of the time variable. Defaults to 'time'.
            range_var (str, optional): Name of the range variable. Defaults to 'range'.
            save_fig (bool, optional): Set to True if you want to save the figure. Defaults to True.
        """
        dataset = self.dataset
        x = dataset[time_var].values
        h = np.round(dataset[range_var].values)
        back_att_arr = dataset[variable_names[0]].T.values
        depol_arr = dataset[variable_names[1]].T.values

        fig, [ax, ax2] = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        lims = [np.max([np.nanmin(back_att_arr), min_value]), np.min([np.nanmax(back_att_arr), max_value])]

        range_limits = get_range_limits(range_limits=range_limits)
        
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

        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')

        if save_fig:
            filepath = filename_to_save(self.dataset, save_fig, suffix='colormesh')
            print(f'saved figure to {filepath}')
            plt.savefig(filepath, bbox_inches='tight', dpi=fig_dpi)
        
        plt.show()
        return  fig, [ax,ax2]

    def plot_histogram(self,
                   variable_1='beta_att_clean',
                   variable_2='linear_depol_ratio_clean',
                   classes_variable=None,
                   variable_1_log=True,
                   variable_2_log=False,
                   count_log=False,
                   colormap=COLOR_MAP,
                   save_fig=True,
                   fig=None,
                   ax=None):

        dataset = self.dataset

        if fig is None:
            fig = plt.figure()

        if ax is None:
            ax = fig.add_subplot(111)

        if variable_2 is None:
            # 1D histogram
            x = dataset[variable_1].values.flatten()
            if classes_variable:
                c = dataset[classes_variable].values.flatten()
                df = pd.DataFrame({variable_1: x, classes_variable: c})
            else:
                c = None
                df = pd.DataFrame({variable_1: x})

            if variable_1_log:
                df[variable_1] = np.log10(df[variable_1])

            if classes_variable:
                sns.set(style="whitegrid")
                sns.histplot(data=df, x=variable_1, hue=classes_variable, multiple="stack",
                            palette=colormap, edgecolor=".3",
                            linewidth=.5, log_scale=count_log, ax=ax)
                ax.set_title(f'1D Histogram of {variable_1}')
                ax.set_xlabel(variable_1)
                ax.set_ylabel('Count (log scale)' if count_log else 'Count')
            else:
                ax.set_title(f'1D Histogram of {variable_1}')
                ax.set_xlabel(variable_1)
                ax.set_ylabel('Count (log scale)' if count_log else 'Count')
                ax.hist(df[variable_1], log=count_log)

        else:
            # 2D histogram
            if (variable_1 is None) or (variable_2 is None):
                raise ValueError('Both variables are required for 2D histogram.')

            if ax is None:
                ax = fig.add_subplot(111)

            x = dataset[variable_1].values.flatten()
            y = dataset[variable_2].values.flatten()
            if variable_1_log:
                x = np.log10(x)
            if variable_2_log:
                y = np.log10(y)
            df = pd.DataFrame({variable_1: x, variable_2: y})

            g = sns.JointGrid(data=df, x=variable_1, y=variable_2, height=8, ratio=4)
            hb = g.ax_joint.hexbin(x, y, gridsize=500, cmap=colormap, bins=count_log)

            cax = plt.gcf().add_axes([1.05, 0.1, 0.05, 0.7])
            cbar = plt.colorbar(hb, cax=cax)

            sns.histplot(data=df, x=variable_1, ax=g.ax_marg_x, color=colormap, cbar_kws={"edgecolor": None})
            sns.histplot(data=df, y=variable_2, ax=g.ax_marg_y, color=colormap, orientation='horizontal', cbar_kws={"edgecolor": None})

        if save_fig:
            plt.savefig(filename_to_save(dataset, save_fig, suffix=f'hist_{variable_1}_vs_{variable_2}'), dpi=300)

        plt.show()

        return fig, ax

    def vertical_profile(self, time_period=None,
                            var_names=['beta_att', 'linear_depol_ratio'],
                            range_limits=[0, 15000],
                            xlabel1='Beta attenuation',
                            xlabel2='linear depol ratio',
                            ylabel='range [m]',
                            title= None,
                            var_xlims=[[1e-7, 1e-4],
                                        [0, 1]],
                            x_scales=['log', 'linear'],
                            plot_colors=['#124E63', '#F6A895'],
                            fig=None,
                            ax=None,
                            save_fig=False,
                            fig_dpi=300):
        """
        Plot profiles of beta attenuation and depolarization ratio respect to height (range).

        Args:
            time_period (str or list, optional): Time period or list of two datetime strings following the format "year-month-day hour:min:sec". Defaults to None.
            var_names (list, optional): Variable names. Defaults to ['beta_att', 'linear_depol_ratio'].
            range_limits (list, optional): Limit of heights to plot. Defaults to [0, 15000].
            xlabel1 (str, optional): X-label description of variable 1. Defaults to 'Beta attenuation'.
            xlabel2 (str, optional): X-label description of variable 2. Defaults to 'Linear depol ratio'.
            ylabel (str, optional): Y-label description of height variable. Defaults to 'range [m]'.
            title (str, optional): Title of the plot. Defaults to None.
            var_xlims (list, optional): X-axis limits for variables. Defaults to [[1e-7, 1e-4], [0, 1]].
            x_scales (list, optional): List of x-axis scales ('log' or 'linear') for variables. Defaults to ['log', 'linear'].
            plot_colors (list, optional): List of plot colors. Defaults to ['#124E63', '#F6A895'].
            ax (matplitlib axis object, optional): Matplotlib axis to use for the plot. Defaults to None.
            save_fig (bool, optional): Whether to save the figure. Defaults to True.
            fig_dpi (int, optional): DPI for saved figure. Defaults to 300.
        Raises:
            ValueError: if time period is not defined as expected

        Returns:
            ax, ax2 (matplotlib axis objects): The primary and secondary axes used in the plot.
        """
        # get dataset
        dataset = self.dataset

        # Handle time_period input
        # Handle time_period input
        if time_period is None:
            time_period = "Entire Period"
        elif isinstance(time_period, str):
            time_period = [time_period]
        elif not isinstance(time_period, list) or len(time_period) != 2:
            raise ValueError("time_period should be a string, a list of two datetime strings, or None.")

        # Select data based on time_period
        if len(time_period) == 2:
            subset = dataset.sel(time=slice(time_period[0], time_period[1])).mean(dim='time')
        else:
            subset = dataset.sel(time=time_period[0], method="nearest")

        # Get time period wished
        # if time_period:
        #     if isinstance(time_period, str):
        #         subset = dataset.sel(time=time_period, method="nearest")
        #     elif isinstance(time_period, list):
        #         if len(time_period) > 2:
        #             raise ValueError(f"Expected max 2 values (start time/date, end time/date) but got {len(time_period)}")
        #         subset = dataset.sel(time=slice(time_period[0], time_period[1])).mean(dim='time')
        #     else:
        #         raise TypeError(f"Expected time_period to be of type str or list, but got {type(time_period)}")
        # else:
        #     subset = dataset.mean(dim='time')

        # Get variables
        beta_atts = subset[var_names[0]]
        lin_depols = subset[var_names[1]]
        heights = subset['range']

        # Create a figure and suplot 
        if fig is None:
            fig = plt.figure(figsize=(4, 6))

        if ax is None:
            ax = fig.add_subplot(111)

        # Plot the first variable with a logarithmic x-axis
        if x_scales[0] == 'log':
            ax.semilogx(beta_atts, heights, label=xlabel1, color=plot_colors[0], alpha=0.8)
        else:
            ax.plot(beta_atts, heights, label=xlabel1, color=plot_colors[0], alpha=0.8)

        ax.set_xlabel(xlabel1, color=plot_colors[0])
        ax.set_ylabel(ylabel)
        ax.set_xlim(var_xlims[0])

        ax.tick_params(axis = 'x', labelcolor = plot_colors[0])
        # Plot the second variable with a linear x-axis on a separate axis
        ax2 = ax.twiny()
        ax2.tick_params(axis='x', labelcolor=plot_colors[1])
        if x_scales[1] == 'log':
            ax2.semilogx(lin_depols, heights, label=xlabel2, color=plot_colors[1], alpha=0.8)
        else:
            ax2.plot(lin_depols, heights, label=xlabel2, color=plot_colors[1], alpha=0.8)
        ax2.set_xlabel(xlabel2, color=plot_colors[1])
        ax2.set_xlim(var_xlims[1])
        
        # set_title:
        if title:
            plt.title(title)
        else:
            plt.title(f'CL61 profiles - {time_period}')
            
        # set height limits
        plt.ylim(range_limits)

        if save_fig:
            filepath = filename_to_save(dataset, save_fig, suffix='vprofile')
            print(f'Saved figure to {filepath}')
            plt.savefig(filepath, bbox_inches='tight', dpi=fig_dpi)
        
        # Show the plot
        if ax == None:
            plt.show()

        return ax, ax2

    def compare_profiles(self, time_period=None, comparison='variable',
                        var_names_1=['beta_att', 'linear_depol_ratio'],
                        var_names_2=['beta_att_clean', 'linear_depol_ratio_clean'],
                        scales=['log', 'lin'],
                        range_limits=[0, 15000],
                        save_fig=True,
                        fig_dpi=400):
        '''
        Creates 2 subplots to compare side by side vertical profiles of beta attenuation and linear depolarisation ratio.

        Args:
            time_period (str or list of str): Time element-s of interest. Expected to be a single str if comparison is not 'time',
            else should be a list of 2 time elements (str) to compare.
            var_names_1 (list): List of the variable names setting the vertical profiles.
            var_names_2 (list): List of the variable names for the 2nd profiles for comparison if comparison is 'variable'.
        '''
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))

        if comparison == 'variable':
            for i, var_names in enumerate([var_names_1, var_names_2]):
                axs[i], ax_twin = self.vertical_profile(
                    time_period=time_period,
                    var_names=var_names,
                    range_limits=range_limits,
                    fig = fig,
                    ax=axs[i]
                )

        elif comparison == 'time':
            if not isinstance(time_period, list) or len(time_period) != 2:
                raise ValueError("If vertical profile comparison in time, time_period is expected to be a list of 2 time strings")

            for i, time in enumerate(time_period):
                axs[i], ax_twin = self.vertical_profile(
                    time_period=time,
                    var_names=var_names_1,
                    range_limits=range_limits,
                    fig = fig,
                    ax=axs[i]
                )

        else:
            raise ValueError("Comparison should be 'variable' or 'time'.")

        if save_fig:
            filepath = filename_to_save(
                dataset=self.dataset.sel(time=time_period, method="nearest"),
                save_name=save_fig,
                suffix='comp_profiles'
            )
            print(f'Saved element to {filepath}')
            plt.savefig(filepath, bbox_inches='tight', dpi=fig_dpi)

        plt.show()

        return
    
# Utility functions
def get_range_limits(range_limits, min_range = 0, max_range = 15000):
    """Checks and returns valid range limits

    Args:
        range_limits (int or list of int): elements of range to set as limit
    
    returns : list of 2 valid int as range limits
    """

    if isinstance(range_limits, int):
        range_limits = [0, max([0, min([range_limits, max_range])])]
    elif isinstance(range_limits, list):
        for id, range_i in enumerate(range_limits):
            range_limits[id] = max([0, min([range_i, max_range])])
        if len(range_limits) == 1:
            range_limits = [0, range_limits[-1]]
        if len(range_limits) > 2:
            print('Did not expect range_limits length > 2 ; taking first 2 values')
            range_limits = range_limits[:2]
    else:
        range_limits = [0, 15000]

    return range_limits



