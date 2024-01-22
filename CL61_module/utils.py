import yaml
import json
import os
import numpy as np
import pandas as pd


def filename_to_save(dataset,
                     save_name,
                     suffix='other',
                     output_folder='..\Outputs'):
    """
    Generate a filename for saving a figure based on dataset information.

    Parameters:
    - dataset: Your dataset object.
    - save_name: str, name to be saved to.
    - suffix: str, optional, additional information to include in the filename.
    - dpi: int, optional, dots per inch for the saved figure (default=300).

    Returns:
    - filename: str, generated filename for saving the figure.
    """
    if isinstance(save_name, str):
        if os.path.dirname(save_name) is None:
            # Only a filename 
            return os.path.join(output_folder, save_name)
        else:
            # already a directory + filename
            return save_name
    elif isinstance(save_name, os.PathLike):
        # Already a path
        return save_name
    else:
        # Determine the default filename based on the first datetime in the dataset
        default_filename = f"{dataset['time'][0].values.astype('datetime64[h]')}_{suffix}"
        return os.path.join(output_folder, default_filename)

def generate_output_folder_name(dataset):
    """
    Generate a name for the output folder based on a datetime and position

    Parameters:
    - dataset: xarray.Dataset
        The input spatio-temporal dataset.

    Returns:
    - output_folder_name: str
        The generated output folder name.
    """
    
    if 'time' not in dataset:
        raise ValueError('Variable time not found in dataset')
    
    # Extract information from the dataset
    first_datetime = dataset.time[0]
    
    folder_name_parts = [f"{first_datetime.dt.strftime('%Y-%m-%d').values}"]

    # Create a folder name based on latitude and longitude
    folder_name_parts.extend([
        f"lat={dataset['latitude'].values:.2f}",
        f"lon={dataset['longitude'].values:.2f}"
        ])

    # Combine parts to create the folder name
    output_folder_name = "_".join(folder_name_parts).replace('.', '_')

    return output_folder_name


def load_config(filepath = None):
    '''Opens and loads config information from given json file at filepath'''

    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), 'config_classification.json')
    
    print(f'Loading config file at {filepath} ')
    with open(filepath, 'r') as file:
        if file is None:
            raise TypeError('File not found at given filepath')
        config_classification = json.load(file)
    return config_classification