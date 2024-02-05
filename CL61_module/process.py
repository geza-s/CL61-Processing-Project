# necessary libraries
import numpy as np
import pandas as pd
import scipy.signal as signal
import xarray as xr


class NoiseProcessor:
    def __init__(self, parent):
        self.CL61_parent = parent
        self.dataset = parent.dataset

    def mask_noise(self, beta_att_window_size=[7, 7], linear_depol_windows_size=[1, 3]):
        """
        Process noise in the data by searching for negative values for beta_attenuation
        and creates new variables with suffix _clean in the dataset and a 'noise_mask'

        Args:
            beta_att_window_size (int or list of int): size of the window for noise removal for beta attenuation coef
            linear_depol_windows_size (int or list of int): size of the window for noise removal for linear
                                                            depolarization ratio
        Returns:
            None
        """
        # Ref to dataset
        dataset = self.dataset

        # Compute non-null masks
        beta_att_non_null_mask = non_noise_windows_mask(dataset,
                                                        variable_name='beta_att',
                                                        window_size=beta_att_window_size,
                                                        analysis_type='non-negative')

        linear_depol_non_null_mask = non_noise_windows_mask(dataset,
                                                            variable_name='linear_depol_ratio',
                                                            window_size=linear_depol_windows_size,
                                                            analysis_type='range')

        final_mask = beta_att_non_null_mask & linear_depol_non_null_mask
        to_interpolate_mask = beta_att_non_null_mask & ~linear_depol_non_null_mask

        print('The results are stored under new variable: beta_att_clean, linear_depol_ratio_clean, noise mask and '
              'to_interpolate_mask')

        # Save result into new variable and assign name/description elements
        self.CL61_parent.dataset['beta_att_clean'] = xr.where(final_mask, dataset['beta_att'], np.nan)
        self.CL61_parent.dataset['beta_att_clean'].attrs['units'] = self.dataset['beta_att'].attrs['units']
        self.CL61_parent.dataset['beta_att_clean'].attrs['long_name'] = "filtered attenuated backscatter coefficient"
        self.CL61_parent.dataset['beta_att_clean'].attrs['original_variable'] = 'attenuated backscatter coefficient'

        self.CL61_parent.dataset['linear_depol_ratio_clean'] = xr.where(final_mask, dataset['linear_depol_ratio'],
                                                                        np.nan)
        self.CL61_parent.dataset['linear_depol_ratio_clean'].attrs['long_name'] = "filtered linear depolarisation ratio"
        self.CL61_parent.dataset['beta_att_clean'].attrs['original_variable'] = 'linear depolarisation ratio'

        self.CL61_parent.dataset['noise_mask'] = xr.DataArray(data=final_mask, dims=['time', 'range'])
        self.CL61_parent.dataset['noise_mask'].assign_attrs(name='noise mask')

        self.CL61_parent.dataset['to_interpolate_mask'] = xr.DataArray(data=to_interpolate_mask, dims=['time', 'range'])
        self.CL61_parent.dataset['to_interpolate_mask'].assign_attrs(long_name='difference in individual noise masks')

        return

    def rolling_window_stats(self, variable_name, stat='mean',
                             time_window_size=5, range_window_size=5):
        """
        Performs rolling window statistics on data array of given variable in dataset.

        Args:
            variable_name (str): variable in dataset
            stat (['mean', 'median', 'std'], optional): statistic. Defaults to 'mean'.
            time_window_size (int, optional): size of window along time dimension. Defaults to 5.
            range_window_size (int, optional): size of window along range. Defaults to 5.

        Raises:
            ValueError: If variable not in dataset

        Returns:
            xarray datarray: datarray of rolling statistics result
        """
        if variable_name not in self.dataset:
            raise ValueError(f"Variable '{variable_name}' not found in the dataset.")

        variable_data = self.dataset[variable_name]

        if stat == 'mean':
            rolling_result = variable_data.rolling(time=time_window_size,
                                                   range=range_window_size,
                                                   min_periods=1, center=True).mean(keep_attrs=True)
        elif stat == 'median':
            rolling_result = variable_data.rolling(time=time_window_size,
                                                   range=range_window_size,
                                                   min_periods=1, center=True).median(keep_attrs=True)
        elif stat == 'std':
            rolling_result = variable_data.rolling(time=time_window_size,
                                                   range=range_window_size,
                                                   min_periods=1, center=True).std(keep_attrs=True)
        else:
            rolling_result = None
            print(f"stat of type {stat} not supported")

        # Save result into array and add relevant attributes
        var_name = f"{variable_name}_roll_{stat}"
        self.dataset[var_name] = rolling_result
        if 'original_variable' in self.dataset[var_name].attrs.keys():
            self.dataset[var_name].attrs['long_name'] = (
                f"({time_window_size},{range_window_size}) rolling {stat} "
                f"{self.dataset[var_name].attrs['original_variable']}"
            )
            self.dataset[var_name].attrs[
                'name'] = f"(rolling {stat} {self.dataset[var_name].attrs['original_variable']}"
        elif 'name' in self.dataset[var_name].attrs.keys():
            self.dataset[var_name].attrs['long_name'] = (
                f"({time_window_size},{range_window_size}) rolling {stat} "
                f"{self.dataset[var_name].attrs['name']}"
            )
            self.dataset[var_name].attrs['name'] = f"(rolling {stat} {self.dataset[var_name].attrs['name']}"

        print(f'Saved the result as variable : {var_name}')

        return


def non_noise_windows_mask(dataset, variable_name, window_size, analysis_type='non-negative', value_range=[0, 1]):
    """
    Checks not only if moving window around element (i.j) has only positive values but if the upper/lower/Right/Left
    windows are non-negative also
    Args:
        dataset (xarray dataset, required): dataset in wich variable_name is contained
        variable_name (str, required): variable name of array on which apply the validity check
        window_size (int, or list of two ints): window size
        analysis_type (str, optional): analysis type to use. Possible values are 'non-negative', 'range'.
                                    Defaults to 'non-negative'.
    Returns
        non_noise_mask (xarray dataArray): Binary array with valid elements as True (1)
    """

    if isinstance(window_size, int):
        window_size = (window_size, window_size)

    if analysis_type == 'non-negative':
        no_noise_mask = rolling_window_all_positive(dataset=dataset, variable_name=variable_name,
                                                    window_size=window_size, border_padding=None).astype(int)
    elif analysis_type == 'range':
        no_noise_mask = rolling_window_in_range_mask(dataset=dataset, variable_name=variable_name,
                                                     window_size=window_size, value_range=value_range,
                                                     border_padding=None).astype(int)
    else:
        raise TypeError("analysis_type should be one of the following ['non-negative', 'range']")

    kernel = create_kernel(window_size)
    non_noise_mask = check_kernel_correlation(no_noise_mask, kernel)

    return non_noise_mask


def rolling_window_all_positive(dataset,
                                variable_name='beta_att',
                                window_size=[5, 5],
                                border_padding=None):
    """
    Checks if all elements in a moving window are positive values

    Parameters:
    - dataset: Xarray dataset of float-type non-null values
    - variable_name: name of the variable in dataset on which to mask
    - window_size: Tuple (rows, cols) specifying the size of the moving window.
    - border_padding: Padding over border transformed to value of border (TO IMPLEMENT)

    Returns:
    - out_mask: 2D xarray boolean array of same shape as input xarray
    """
    if type(window_size) == int:
        window_size = (window_size, window_size)

    # boolean array (as int 0,1) if value is >0
    positive_values = (dataset[variable_name] > 0).astype(int)

    # Rolling window sum of all boolean as integer value
    min_elements_for_window = max([window_size[0] // 2 * window_size[1] // 2, 1])
    positive_values_window_sum = positive_values.rolling(time=window_size[0], range=window_size[1], center=True,
                                                         min_periods=min_elements_for_window).reduce(np.sum)

    # Check if all elements in window as True (in range)
    window_element_numbers = window_size[0] * window_size[1]
    mask_positive_value_windows = positive_values_window_sum == window_element_numbers

    return mask_positive_value_windows


def rolling_window_in_range_mask(dataset,
                                 variable_name='linear_depol_ratio',
                                 window_size=[5, 5],
                                 value_range=[0, 1],
                                 border_padding=None):
    """
    Checks if all elements in a moving window are inside given range

    Parameters:
    - dataset: Xarray dataset of continuous and non-null values
    - variable_name: name of the variable in dataset on which to mask
    - window_size: Tuple (rows, cols) specifying the size of the moving window.
    - value_range: Range of values to mask as "in range"
    - border_padding: Padding over border transformed to value of border

    Returns:
    - out_mask: 2D xarray boolean array of same shape as input xarray
    """
    if isinstance(window_size, int):
        window_size = (window_size, window_size)

    # boolean array (as int 0,1) if in range
    in_range = np.logical_and(dataset[variable_name] > value_range[0], dataset[variable_name] < value_range[1]).astype(
        int)

    # Rolling window sum of all boolean in range as integer value
    min_elements_for_window = max([window_size[0] // 2 * window_size[1] // 2, 1])
    in_range_window_sum = in_range.rolling(time=window_size[0], range=window_size[1], center=True,
                                           min_periods=min_elements_for_window).reduce(np.sum)

    # Check if all elements in window as True (in range)
    window_element_numbers = window_size[0] * window_size[1]
    mask_in_range = in_range_window_sum == window_element_numbers

    return mask_in_range


def create_kernel(window_size):
    """
    Create a diamond kernel to encompass all possible windows in which element (i,j) could be
    """
    kernel = np.zeros(window_size, dtype=int)
    center_time, center_range = np.array(window_size) // 2
    size_time, size_range = window_size
    time_values = np.arange(0, size_time)
    range_values = np.arange(0, size_range) / size_range * size_time

    time_mesh, range_mesh = np.meshgrid(time_values, range_values)
    kernel = np.array(np.abs(time_mesh - center_time) + np.abs(range_mesh - center_time) <= center_time).astype(int)

    return kernel


def check_kernel_correlation(boolean_mask_array, kernel):
    """
    Correlate a boolean mask array with the given kernel and check if the result presents some non-null overlap.
    Args:
        boolean_mask_array: boolean mask to check
        kernel: kernel to correlate with.
    Returns:
        non_noise_mask: boolean mask of valid data points
    """
    correlated_result = signal.correlate2d(boolean_mask_array, kernel.T, mode='same', boundary='symm', fillvalue=0)
    non_noise_mask = correlated_result > 0

    return non_noise_mask
