import yaml
import json
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
        default_filename = f"{dataset['time'][0].values.astype('datetime64[h]')}_{suffix}"
        return os.path.join(output_folder, default_filename)


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
