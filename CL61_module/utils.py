import yaml
import os

def load_config(filepath = '../CL61_module/config_classification.yml'):
    print(f'Loading config file at {filepath} ')
    with open(filepath, 'r') as file:
        if file == None:
            raise TypeError('File not found at given filepath')
        config_classification = yaml.safe_load(file)
    return config_classification