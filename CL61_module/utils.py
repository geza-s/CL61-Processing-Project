import yaml
import os
import numpy as np
import pandas as pd


def filename_to_save(dataset,
                     save_name,
                     suffix='other',
                     output_folder='..\Outputs'):
    '''
    Finds the path and appropriate name for saving file based on period of interest of dataset
    '''
    if isinstance(save_name, str):
        return os.path.join(output_folder, save_name)
    else:
        # Determine the default filename based on the first datetime in the dataset
        default_filename = f"{dataset['time'].min().values.astype('datetime64[D]')}_{suffix}"
        return os.path.join(output_folder, default_filename)


def load_config(filepath='../CL61_module/config_classification.yml'):
    print(f'Loading config file at {filepath} ')
    with open(filepath, 'r') as file:
        if file is None:
            raise TypeError('File not found at given filepath')
        config_classification = yaml.safe_load(file)
    return config_classification
