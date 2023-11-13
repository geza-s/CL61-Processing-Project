import numpy as np
import pandas as pd
import scipy.signal as signal
import xarray as xr

def rolling_window_all_positive(dataset,
                                variable_name = 'beta_att',
                                window_size = [5,5],
                                border_padding = None):
    """
    Checks if all elements in a moving window are positive values

    Parameters:
    - dataset: Xarray dataset of float-type non null values
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
    min_elements_for_window = max([window_size[0]//2*window_size[1]//2,1])
    positive_values_window_sum = positive_values.rolling(time=window_size[0], range = window_size[1], center=True, min_periods=min_elements_for_window).reduce(np.sum)
    
    # Check if all elements in window as True (in range)
    window_element_numbers = window_size[0]*window_size[1]
    mask_positive_value_windows = positive_values_window_sum==window_element_numbers

    return mask_positive_value_windows


def rolling_window_in_range_mask(dataset,
                             variable_name = 'linear_depol_ratio',
                             window_size = [5,5],
                             value_range = [0,1],
                             border_padding = None):
    """
    Checks if all elements in a moving window are inside given range

    Parameters:
    - dataset: Xarray dataset of continuous and non null values
    - variable_name: name of the variable in dataset on which to mask
    - window_size: Tuple (rows, cols) specifying the size of the moving window.
    - value_range: Range of values to mask as "in range"
    - border_padding: Padding over border transformed to value of border

    Returns:
    - out_mask: 2D xarray boolean array of same shape as input xarray
    """
    if type(window_size) == int:
        window_size = (window_size, window_size)

    # boolean array (as int 0,1) if in range
    in_range = np.logical_and(dataset[variable_name] > value_range[0], dataset[variable_name] < value_range[1]).astype(int)

    # Rolling window sum of all boolean in range as integer value
    min_elements_for_window = max([window_size[0]//2*window_size[1]//2,1])
    in_range_window_sum = in_range.rolling(time=window_size[0], range = window_size[1], center=True, min_periods=min_elements_for_window).reduce(np.sum)
    
    # Check if all elements in window as True (in range)
    window_element_numbers = window_size[0]*window_size[1]
    mask_in_range = in_range_window_sum==window_element_numbers
    
    # Save result into new variable clean
    #new_variable_name = variable_name + '_clean'
    #dataset[new_variable_name] = xr.where(mask_in_range, dataset[variable_name], np.nan)

    return mask_in_range

def non_noise_windows_mask(dataset, variable_name, window_size, analysis_type = 'non-negative', value_range=[0,1]):
    '''
    Checks not only if moving window around element (i.j) has only positive values
    but if the upper/lower/Right/Left windows are non negative also
    ''' 
    
    if type(window_size) == int:
        window_size = (window_size, window_size)
    
    if analysis_type == 'non-negative':
        no_noise_mask = rolling_window_all_positive(dataset=dataset, variable_name=variable_name,
                                                     window_size=window_size, border_padding=None).astype(int)
    elif analysis_type  == 'range':
        no_noise_mask = rolling_window_in_range_mask(dataset=dataset, variable_name=variable_name,
                                                      window_size=window_size, value_range=value_range, border_padding=None).astype(int)
    else:
        raise TypeError("analysis_type should be one of the following ['non-negative', 'range']")
    
    # Create an empty array of zeros
    kernel = np.zeros(window_size, dtype=int)

    # Create a losange kernel to encompass all possible windows in wich element (i,j) could be
    center_time, center_range = np.array(window_size) // 2
    size_time, size_range = window_size
    T = np.arange(0, size_time)
    R = np.arange(0, size_range)/size_range * size_time  #Crucial to adapt range to threshold with time
    Jt, Ir = np.meshgrid(T,R)
    kernel = np.array(np.abs(Jt - center_time) + np.abs(Ir - center_time) <= center_time).astype(int)
    
    # Correlate with kernel
    correlated_result = signal.correlate2d(no_noise_mask, kernel.T, mode='same', boundary='symm', fillvalue=0)
    
    # Take as non-noise if at least one of the kernel elements (refering to a window of elements) is non noise
    #non_noise_mask = correlated_result<np.sum(kernel)
    non_noise_mask = correlated_result>0

    return non_noise_mask