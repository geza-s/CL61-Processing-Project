# File management
import os

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
from .utils import filename_to_save, load_config

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
    
    def __init__(self, dataset = None, parent = None):
        # Initialize values from input so It can be called indep of main module
        self.dataset = dataset
        self.parent = parent
        # If called from a parent class (CL61 module main class)
        if parent is not None:
            self.parent = parent
            self.dataset = parent.dataset


    def show_timeserie(self,
                        variable_names=['beta_att', 'linear_depol_ratio'],
                        value_ranges=[[1e-7, 1e-4],[0,1]],
                        range_limits=None,
                        log_scale=[True, False],
                        color_map="cmc.batlow",
                        cbar_labels = None,
                        fig = None, axs = None,
                        save_fig = False,
                        fig_dpi = 300,
                        **kwargs):
        
        # Slice data based on range
        range_limits = get_range_limits(range_limits)
        subset = self.dataset.sel(range = slice(*range_limits))

        # Check variables to plot
        if isinstance(variable_names, str):
            variable_names = [variable_names]
        # Number of variables
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

            if log_scale[i]:
                vmin = np.log10(vmin) if vmin != None  else vmin
                vmax = np.log10(vmax) if vmax != None  else vmax
                im = np.log10(subset[var_name]).plot.imshow(x='time', y='range', vmin=vmin, vmax=vmax, ax = ax, cmap=color_map)
            else:
                im = subset[var_name].plot.imshow(x='time', y='range', vmin=vmin, vmax=vmax, ax = ax, cmap = color_map)

            cbar = im.colorbar
            if cbar_labels is None:
                cbar.set_label(var_name[i])
            else:
                if isinstance(cbar_labels, str):
                    cbar_labels = [cbar_labels]
                cbar.set_label(cbar_labels[i])
            
            ax.set_ylabel('range [m]')
            ax.set_title('')

        if save_fig:
            filepath = filename_to_save(self.dataset, save_fig, suffix='colormesh')
            print(f'saved figure to {filepath}')
            plt.savefig(filepath, bbox_inches='tight', dpi=fig_dpi)

        return axs
       
    def show_cloud_base_heights(self, range_limits = [0, 15000], underlying_variable = 'beta_att',
                                ax = None, colormap = cmc.batlow, save_fig = False,
                                figsize = (12, 5), figdpi = 300):
        
        # Slice data based on range
        subset = self.dataset.sel(range = slice(*range_limits))

        # Create plot if not given
        if ax is None:
            fig, ax = plt.subplots(figsize = figsize)
        
        # Plot cloud base heights
        cloud_plot = subset['cloud_base_heights'].plot.scatter(ax = ax, marker = '_',
                                                                color ='#48013b', alpha  = 0.7,
                                                                label = 'cloud base heights')
        
        # Plot underlying variable (beta attenuation by default)
        if underlying_variable in ('beta_att', 'beta_att_clean'):
            im = np.log10(subset[underlying_variable].T).plot.imshow(ax = ax, cmap = colormap, vmin = -8, vmax = -3)
            cbar = im.colorbar
            cbar.set_label('$Log_{10}$ attenuated backscatter \n coefficient  [$m^{-1}~sr^{-1}$]')
        elif underlying_variable in ('linear_depol_ratio', 'linear_depol_ratio_clean'):
            im = subset[underlying_variable].T.plot.imshow(ax = ax, cmap = colormap, vmin = 0, vmax = 1)
            cbar = im.colorbar
            cbar.set_label('Linear depolarization ratio')
        ax.legend(handles=[cloud_plot], loc = 'right', bbox_to_anchor = (1, 1))
        
        ax.set_ylabel('range [m]')
        ax.set_title("")
        
        plt.show()
        
        return ax
        
    # def colormesh(self, variable_names=['beta_att', 'linear_depol_ratio'],
    #                     min_value=1e-7,
    #                     max_value=1e-4,
    #                     range_limits=None,
    #                     scales=['log', 'linear'],
    #                     time_var='time',
    #                     range_var='range',
    #                     color_map=COLOR_MAP_NAME,
    #                     save_fig = False,
    #                     fig_dpi = 300,
    #                     fig = None, axs = None,
    #                     **kwargs):
    #     """
    #     Plot CL61 data as a colormesh.

    #     Args:
    #         dataset (_type_): The dataset to be plotted.
    #         variable_names (list, optional): Names of the variables to be plotted.. Defaults to ['beta_att', 'linear_depol_ratio'].
    #         min_value (_type_, optional):Minimum value for the color scale.. Defaults to 1e-7.
    #         max_value (_type_, optional): Maximum value for the color scale.. Defaults to 1e-4.
    #         range_limits (int or list , optional): max_range or [min_range, max_range] for the y-axis (optional upper limit).. Defaults to None.
    #         color_map (_type_, optional): Matplotlib colormap name. (Default see file).
    #         scales (list, optional): to specify the color scales. Defaults to ['log', 'linear'].
    #         time_var (str, optional): Name of the time variable. Defaults to 'time'.
    #         range_var (str, optional): Name of the range variable. Defaults to 'range'.
    #         save_fig (bool, optional): Set to True if you want to save the figure. Defaults to True.
    #     """
    #     dataset = self.dataset
    #     x = dataset[time_var].values
    #     h = np.round(dataset[range_var].values)
    #     back_att_arr = dataset[variable_names[0]].T.values
    #     depol_arr = dataset[variable_names[1]].T.values

    #     fig, [ax, ax2] = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    #     lims = [np.max([np.nanmin(back_att_arr), min_value]), np.min([np.nanmax(back_att_arr), max_value])]

    #     range_limits = get_range_limits(range_limits=range_limits)
        
    #     if scales[0] == 'log':
    #         cax = ax.pcolormesh(x, h, back_att_arr, axes=ax, shading='nearest', cmap=color_map,
    #                             norm=colors.LogNorm(vmin=lims[0], vmax=lims[1]))
    #     else:
    #         cax = ax.pcolormesh(x, h, back_att_arr, axes=ax, shading='nearest', cmap=color_map, vmin=0, vmax=1)
        
    #     cbar = fig.colorbar(cax)
    #     cbar.ax.get_yaxis().labelpad = 15
    #     cbar.ax.set_ylabel(rf'{variable_names[0]}', rotation=90)
    #     ax.set_ylabel('Range (m)')
    #     # ax.set_xlabel('time')
    #     ax.set_ylim(range_limits)

    #     if scales[1] == 'log':
    #         cax2 = ax2.pcolormesh(x, h, depol_arr, shading='nearest', cmap=color_map,
    #                             norm=colors.LogNorm(vmin=lims[0], vmax=lims[1]))
    #     else:
    #         cax2 = ax2.pcolormesh(x, h, depol_arr, shading='nearest', cmap=color_map, vmin=0, vmax=1)

    #     cbar = fig.colorbar(cax2)
    #     cbar.ax.get_yaxis().labelpad = 15
    #     cbar.ax.set_ylabel(rf'{variable_names[1]}', rotation=90)
        
    #     ax2.set_ylabel('Range (m)')
    #     ax2.set_xlabel('time')
    #     ax2.set_ylim(range_limits)

    #     ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right')

    #     if save_fig:
    #         filepath = filename_to_save(self.dataset, save_fig, suffix='colormesh')
    #         print(f'saved figure to {filepath}')
    #         plt.savefig(filepath, bbox_inches='tight', dpi=fig_dpi)
        
    #     plt.show()
    #     return  fig, [ax,ax2]

    def histogram_1d(self, variable = 'beta_att_clean', classes_variable=None,
            variable_logscale=True, count_log=False, colormap=COLOR_MAP,
            save_figure=True, fig=None, ax=None):
        """
        Plot a 1D histogram for the specified variable in the dataset.

        Parameters:
        - dataset: pandas.DataFrame, required, dataset containing the variable for plotting.
        - variable: str, required, name of the variable to plot the histogram for.
        - classes_variable: str, optional, variable used for coloring the histogram if applicable.
        - variable_logscale: bool, optional, if True, apply logarithmic scale to the variable.
        - count_log: bool, optional, if True, the count axis is displayed in log scale.
        - colormap: str or list, optional, colormap to use for plotting.
        - save_figure: bool, optional, if True, save the plotted figure.
        - fig: matplotlib.figure.Figure, optional, existing figure for plotting.
        - ax: matplotlib.axes._axes.Axes, optional, existing axes for plotting.

        Returns:
        - fig: matplotlib.figure.Figure, the plotted figure.
        - ax: matplotlib.axes._axes.Axes, the plotted axes.
        """

        # dataset ref
        dataset = self.dataset
        check_variables(dataset, variable)

        # Check inputs and adapt if possible
        if isinstance(variable, str):
            variables = [variable]
        if isinstance(variable_logscale, bool):
            variable_logscales = [variable_logscale]
        if len(variable_logscales) < len(variables):
            raise(ValueError("variable_logscale should be specified for each variable given (same length)"))

        # 1D histogram
        variable_1 = variables[0]
        x = dataset[variable_1].values.flatten()

        if fig is None:
            fig = plt.figure(figsize=(8, 5))

        if ax is None:
            ax = fig.add_subplot(111)

        if classes_variable:
            c = dataset[classes_variable].values.flatten()
            df = pd.DataFrame({variable_1: x, classes_variable: c})
        else:
            c = None
            df = pd.DataFrame({variable_1: x})

        if variable_logscale[0]:
            df[variable_1] = np.log10(df[variable_1])

        if classes_variable:
            sns.set(style="whitegrid")
            sns.histplot(
                data=df, x=variable_1, hue=classes_variable, multiple="stack", palette=colormap, edgecolor=".3", linewidth=.5,
                log_scale=count_log, ax=ax
            )
            ax.set_title(f"1D Histogram of {variable_1}")
            ax.set_xlabel(variable_1)
            ax.set_ylabel("Count (log scale)" if count_log else "Count")
        else:
            ax.set_title(f"1D Histogram of {variable_1}")
            ax.set_xlabel(variable_1)
            ax.set_ylabel("Count (log scale)" if count_log else "Count")
            sns.histplot(data=df, x=variable_1, edgecolor=".3", linewidth=.5, log_scale=count_log, ax=ax)

        if save_figure:
            plt.savefig(filename_to_save(dataset, save_figure, suffix=f"hist_{variable_1}"), dpi=300)

        return fig, ax

    def histogram_2d(self,
                    variable_1 = 'beta_att_clean',
                    variable_2='linear_depol_ratio_clean',
                    variable_logscales=[True, False],
                    count_log=True,
                    save_figure=True,
                    colormap=COLOR_MAP):
        """
        Plot a 2D histogram for the specified variables in the dataset.

        Parameters:
        - dataset: pandas.DataFrame, required, dataset containing the variables for plotting.
        - variable_1: str, required, name of the first variable to plot the histogram for.
        - variable_2: str, required, name of the second variable to plot the histogram for.
        - variable_logscales: list of bool, optional, list indicating whether to apply log scale to each variable.
        - count_log: bool, optional, if True, the count axis is displayed in log scale.
        - colormap: str or list, optional, colormap to use for plotting.
        - save_figure: bool, optional, if True, save the plotted figure.
        - fig: matplotlib.figure.Figure, optional, existing figure for plotting.
        - ax: matplotlib.axes._axes.Axes, optional, existing axes for plotting.

        Returns:
        - fig: matplotlib.figure.Figure, the plotted figure.
        - ax: matplotlib.axes._axes.Axes, the plotted axes.
        """
        # Dataset reference and validity check
        dataset = self.dataset
        check_variables(dataset, variables=[variable_1, variable_2]) #raise understandible error if var not in dataset         
        
        # Flatten values for binning (histograms)
        x = dataset[variable_1].values.flatten()
        y = dataset[variable_2].values.flatten()

        # apply scale transformation
        if variable_logscales[0]:
            x = np.log10(x)
        if variable_logscales[1]:
            y = np.log10(y)

        df = pd.DataFrame({variable_1: x, variable_2: y})

        g = sns.JointGrid(
            data=df, x=variable_1, y=variable_2, height=7, ratio=4
        )
        if count_log:
            hb = g.ax_joint.hexbin(x, y, gridsize=300, cmap=colormap, bins="log")
        else:
            hb = g.ax_joint.hexbin(x, y, gridsize=300, cmap=colormap)

        cax = plt.gcf().add_axes([1.05, 0.1, 0.05, 0.7])
        cbar = plt.colorbar(hb, cax=cax)

        sns.histplot(
            data=df,
            x=x,
            ax=g.ax_marg_x,
            cbar_kws={"edgecolor": None},
            color='#03275C',
            element="step"
        )
        sns.histplot(
            y=y,
            ax=g.ax_marg_y,
            cbar_kws={"edgecolor": None},
            color='#03275C',
            element="step"
        )

        if save_figure:
            plt.savefig(
                filename_to_save(dataset, save_figure, suffix=f"hist_{variable_1}_{variable_2}"), dpi=300
            )

        return g

    def histogram(self,
                        variables=['beta_att_clean', 'linear_depol_ratio_clean'],
                        classes_variable=None,
                        variable_logscales=[True, False],
                        count_log=False,
                        colormap=COLOR_MAP,
                        save_figure=True,
                        fig=None,
                        ax=None):
        """
        Plot 1D or 2D histograms for specified variables in the dataset.

        Parameters:
        - variables: list of str, required, list of variables for which histograms will be plotted.
        - classes_variable: str, optional, variable used for coloring if 1D histogram includes classes.
        - variable_logscales: list of bool, optional, list indicating whether to apply log scale to each variable.
        - count_log: bool, optional, if True, the count axis is displayed in log scale.
        - colormap: str or list, optional, colormap to use for plotting.
        - save_figure: bool, optional, if True, save the plotted figure.
        - fig: matplotlib.figure.Figure, optional, existing figure for plotting.
        - ax: matplotlib.axes._axes.Axes, optional, existing axes for plotting.

        Returns:
        - fig: matplotlib.figure.Figure, the plotted figure.
        - ax: matplotlib.axes._axes.Axes, the plotted axes.
        """
        
        # dataset ref
        dataset = self.dataset

        # Check inputs and adapt if possible
        if isinstance(variables, str):
            variables = [variables]
        if isinstance(variable_logscales, bool):
            variable_logscales =[variable_logscales]
        if len(variable_logscales)<len(variables):
            raise(ValueError("variable_logscale should be specified for each variable given (same length)"))

        if len(variables)==1:
            # 1D histogram
            variable_1 = variables[0]
            x = dataset[variable_1].values.flatten()
            
            if fig is None:
                fig = plt.figure(figsize=(8,5))

            if ax is None:
                ax = fig.add_subplot(111)
            
            if classes_variable:
                c = dataset[classes_variable].values.flatten()
                df = pd.DataFrame({variable_1: x, classes_variable: c})
            else:
                c = None
                df = pd.DataFrame({variable_1: x})

            if variable_logscales[0]:
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
                sns.histplot(data=df, x=variable_1, edgecolor=".3",
                            linewidth=.5, log_scale=count_log, ax=ax)

            if save_figure:
                plt.savefig(filename_to_save(dataset, save_figure, suffix=f'hist_{variable_1}'), dpi=300)
            
        elif len(variables)==2:
            # 2D histogram
            variable_1, variable_2 = variables

            # Flatten values for binning (histograms)
            x = dataset[variable_1].values.flatten()
            y = dataset[variable_2].values.flatten()
            
            # apply scale transformation
            if variable_logscales[0]:
                x = np.log10(x)            
            if variable_logscales[1]:
                y = np.log10(y)
                
            df = pd.DataFrame({variable_1: x, variable_2: y})

            g = sns.JointGrid(data=df, x=variable_1, y=variable_2, height=7, ratio=4)
            hb = g.ax_joint.hexbin(x, y, gridsize=300, cmap=colormap, bins='log')
                
            cax = plt.gcf().add_axes([1.05, 0.1, 0.05, 0.7])
            cbar = plt.colorbar(hb, cax=cax)

            sns.histplot(data=df, x=variable_1, ax=g.ax_marg_x, cbar_kws={"edgecolor": None})
            sns.histplot(y=y, ax=g.ax_marg_y, cbar_kws={"edgecolor": None})

            if save_figure:
                plt.savefig(filename_to_save(dataset, save_figure, suffix=f'hist_{variable_1}_vs_{variable_2}'), dpi=300)
        else:
            raise(ValueError("More than 2 variables not supported"))



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

        # Take time subset:
        subset = get_xarray_subset_time_range(time_period=time_period, dataset=self.dataset)

        # Get variables
        if 'time' in subset.dims:
            beta_atts = subset[var_names[0]].mean(dim='time', skipna=True)
            lin_depols = subset[var_names[1]].mean(dim='time', skipna=True)
        else:
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
            filepath = filename_to_save(subset, save_fig, suffix='vprofile')
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
                    var_names=var_names,
                    time_period=time_period,
                    xlabel1=var_names[0],
                    xlabel2=var_names[1],
                    range_limits=range_limits,
                    fig = fig,
                    ax=axs[i]
                )

        elif comparison == 'time':
            if not isinstance(time_period, list) or len(time_period) != 2:
                raise ValueError("If vertical profile comparison in time, time_period is expected to be a list of 2 time strings")

            for i, time in enumerate(time_period):
                axs[i], ax_twin = self.vertical_profile(
                    var_names=var_names_1,
                    time_period=time,
                    xlabel1=var_names_1[0],
                    xlabel2=var_names_1[1],
                    range_limits=range_limits,
                    fig = fig,
                    ax=axs[i]
                )

        else:
            raise ValueError("Comparison should be 'variable' or 'time'.")

        if save_fig:
            filepath = filename_to_save(
                dataset=get_xarray_subset_time_range(time_period=time_period, dataset=self.dataset),
                save_name=save_fig,
                suffix='comp_profiles'
            )
            print(f'Saved element to {filepath}')
            plt.savefig(filepath, bbox_inches='tight', dpi=fig_dpi)

        plt.show()

        return
    
    def show_classified_timeserie(self, classified_variable = 'classified_clusters',
                                ylims=[0, 10000], fig=None, ax=None, save_fig = False, title = None):
        '''Plots the classifed array with id corresponding to class naem as given in config file'''

        config = self.parent.classification.config
        
        # Get necessary values from the config file for colors
        category_colors = [category['color'] for category in config['classes']]
        category_ids = [category['class_id'] for category in config['classes']]
        classification_cmap = ListedColormap(category_colors)
        num_categories = len(config['classes'])

        if (fig is None) or (ax is None):
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        # Create a colored mesh plot using the custom colormap
        plot = ax.pcolormesh(self.dataset['time'], self.dataset['range'], self.dataset[classified_variable].T,
                            cmap=classification_cmap, vmin=0, vmax=num_categories, shading='nearest')

        # Add a colorbar with discrete color labels
        cbar = fig.colorbar(plot, cmap=classification_cmap)
        cbar.set_ticks([i+0.5 for i in category_ids])
        cbar.set_ticklabels([f"{category['class_id']}: {category['class_name']}" for category in config['classes']])

        # Set labels for x and y axes
        ax.set_ylim(ylims)
        ax.set_xlabel('Time')
        ax.set_ylabel('Range [m]')

        # Set the title of the plot
        if title:
            ax.set_title(title)
        # Rotate time labels (xticks)
        ax.tick_params(axis='x', rotation=45, which='major')

        if save_fig:
            filepath = filename_to_save(
                dataset=self.dataset,
                save_name=save_fig,
                suffix='classified'
            )
            print(f'Saved element to {filepath}')
            plt.savefig(filepath, bbox_inches='tight', dpi=300)


        # Show the plot
        plt.show()
        
        return fig, ax

# Utility functions ------------------------------------------------------------------------------

def get_xarray_subset_time_range(time_period, dataset):
    ''' Checks for the given time period and returns a subset corresponding to the time period.
    Arguments:
    time_period: str or a list of 2 str referring to time.
    dataset: xarray dataset with 'time' dimension
    '''
    # Validate the input dataset
    if 'time' not in dataset.dims:
        raise ValueError("The dataset must have a 'time' dimension.")

    # Handle time_period input
    time_range = [dataset['time'][0].values, dataset['time'][-1].values]

    if time_period is None:
        return dataset
    
    elif isinstance(time_period, str):
        time_period = [time_period]
    
    elif not isinstance(time_period, list) or len(time_period) > 2:
        raise ValueError("time_period should be a string, a list of max two datetime strings, or None.")

    # Check if time is in the time range
    if len(time_period) == 2:
        start_time, end_time = np.datetime64(time_period[0]), np.datetime64(time_period[1])
        if start_time < time_range[0] or end_time > time_range[1]:
            raise ValueError("Time period {} to {} is outside the dataset's time range {} to {}."
                             .format(start_time, end_time, np.datetime_as_string(time_range[0], unit='s'), np.datetime_as_string(time_range[1], unit='s')))
    elif len(time_period) == 1:
        target_time = np.datetime64(time_period[0])
        if target_time < time_range[0] or target_time > time_range[1]:
            raise ValueError("Time point {} is outside the dataset's time range {} to {}."
                             .format(target_time, np.datetime_as_string(time_range[0], unit='s'), np.datetime_as_string(time_range[1], unit='s')))

    # Select data based on time_period
    if len(time_period) == 2:
        subset = dataset.sel(time=slice(time_period[0], time_period[1]))
    else:
        subset = dataset.sel(time=time_period[0], method="nearest")

    return subset

def check_variables(dataset, variables):
    if isinstance(variables, str):
        variables = [variables]
    elif not isinstance(variables, list):
        raise ValueError("Variables should be a string or a list")

    for variable in variables:
        if variable not in dataset:
            raise KeyError(f"'{variable}' does not exist in the dataset.")
    return

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

