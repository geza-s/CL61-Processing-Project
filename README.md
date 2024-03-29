# Vaisala CL61 Ceilometer Data Analysis Project

## Overview

This project is a master's student project aimed at analyzing and processing data collected from a Vaisala CL61 ceilometer. The project focuses on processing and interpreting ceilometer backscattering data to explore the potential of last-generation ceilometers to characterize atmospheric observations. This could potentially be beneficial for various applications, such as weather monitoring, climate research, or environmental studies.

## Paper abstract

  This semester's project aims to explore the potential of the Vaisala CL61 ceilometer with polarization capacity for characterizing atmospheric observations. To achieve this, a Python pipeline was developed to classify atmospheric targets. The noise was filtered out from the data, enabling an efficient application of K-means clustering. Each cluster was classified by fixing thresholds on attenuated backscatter and linear depolarization ratio. The results show differentiation between liquid and solid scattering targets as well as the identification of aerosol plumes, demonstrating the extended use of ceilometers through their polarization capabilities.

## Project Goals

- Understand and interpret data from the Vaisala CL61 ceilometer.

- Develop a Python module for data analysis and processing.

- Generate meaningful insights and visualizations from the data.

- Create a structured project that facilitates further research and experimentation.

## Project Structure

The project folder is structured as follows:

- `CL61_module/`: This directory contains the Python module responsible for processing and analyzing data from the Vaisala CL61 ceilometer. It includes relevant configuration files and module-specific code.

- `Data_samples/`: Sample of data obtained from the Vaisala CL61 ceilometer are stored here. These data files are used as input for analysis and processing in the Python module.

- `Outputs/`: Output files generated by the analysis process are saved here. This can include visualizations, reports, and processed data for further use.

- `tests/`: This directory contains Jupyter Notebooks or Python scripts used for testing and analyzing the module's functionality. It serves as a workspace for experimentation and research.

## Module Structure

The module is organized into classes that handle different aspects of data processing and visualization. Here's an overview of the main classes:

### `CL61Processor`

The `CL61Processor` class is the core of the module which wraps around the loading of the dataset and manage the processing. It has three main subclasses, each accessible through different methods.

#### Subclasses

1. **Plotting (`.plot`):**
    - The `Plotting` subclass handles the generation of plots based on processed data.
    - Example usage:
      ```python
      cl61_processor = CL61Processor(...)
      cl61_processor.plot.show_timeserie()
      ```

2. **Noise Processing (`.process_noise`):**
    - The `NoiseProcessing` subclass focuses on noise processing within the data.
    - Example usage:
      ```python
      cl61_processor = CL61Processor(...)
      cl61_processor.process_noise.mask_noise()
      ```

3. **Classification (`.classification`):**
    - The `Classification` subclass is responsible for data classification (clustering and classification)
    - Example usage:
      ```python
      cl61_processor = CL61Processor(...)
      cl61_processor.classification.Kmeans_clustering()
      ```

## Getting Started

To get started with this project, follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies by running `pip install -r requirements.txt`.

3. Explore the `CL61_module/` directory to understand the module's functionality and configuration.

4. ### Data Analysis and Processing
    Two options are suggested:

   a) Follow the proposed usage presented in the jupyter notebook scripts `main_analysis.ipynb` for dataset stored in local folder or on server. More usage examples are given in the `tests/` directory (see f.ex. `tests/main_paper_results.ipynb`)

    The basic use is with the following example:
    ```python
    import CL61_module as mCL61
    start_data = '2023-02-22 00:00:00'
    end_data = '2023-02-23 00:00:00'
    data_folder = "/path/to/your/data"

    feb22 = mCL61.CL61Processor(folder_path=data_folder,
                                     start_datetime=start_data, end_datetime=end_data,
                                     load_to_memory=False, transfer_files_locally=False)
    
    #automated processing:
    feb22.auto_process(range_limits = [0, 10000])

    ```
    - Use `load_to_memory=True` to improve processing time
    - And  `transfer_files_locally=True` transfers files to a local temporary folder improving significantly the processing time (to use when data is on server)

   b) (Shortcut) Use the module directly from command prompt as following:

    Make sure to be in the module directory ('.../CL61_module/').

      ```bash
      python module.py --data-folder "/path/to/your/data" --start-time "YYYY-MM-DD HH:mm:ss" --end-time "YYYY-MM-DD HH:mm:ss" --range [min range, max range]
      ```

      #### Options:

      - `--data-folder`: Path to the data folder. Default is `X:\common\03_Experimental\Instruments\EERL_instruments\5. Remote sensing\CL61_Ceilometer\Data`.

      - `--start-time`: Start date and time in the format `YYYY-MM-DD HH:mm:ss`. Default is `2023-02-22 00:00:00`.

      - `--end-time`: End date and time in the format `YYYY-MM-DD HH:mm:ss`. Default is `2023-02-23 00:00:00`.

      - `--min-range`: Measurement range (height) minimum limit `min_range` for plots. Can go from 0 to 15000 m. Default is `0`

      - `--max-range`: Measurement range (height) maximum limit `max_range` for plots. Can go from 0 to 15000 m. Default is `15000`

      - `--load-to-memory`: Flag to indicate whether to load data into memory. Default is `False`.

      - `--transfer-files-locally`: Flag to indicate whether to transfer files into a local temp folder. Default is `False`.

      #### Example:

      ```bash
      python module.py --data-folder "../Data_samples" --start-time "2023-02-22 12:00:00" --end-time "2023-02-23 12:00:00" --range [500,10000]
      ```

   This example will process data from the specified data folder, starting from the given start time to the end time, and within the specified height range.

## Configuration
This project relies on a configuration file (`CL61_module\config_classification.json`) to define thresholds for classification. You can customize the thresholds and classes by following the established structure and variable names:
```json
{
    "classes": [
      {
        "class_id": 1,
        "class_name": "Smoke/Pollution/Marine",
        "beta_attenuation_range": [1e-6, 5e-6],
        "linear_depolarisation_ratio_range": [0, 0.1],
        "color": "#a9cac4"
      },
      ...
    ]
  }
```

### Class Definitions

- **`class_id` (integer):** An identifier to reference each data point or cluster to its related class. It must be unique, and consider starting with lower integers for better plotting.

- **`class_name` (string):** A descriptive name for the class (used in plots)

- **`beta_attenuation_range` (list of floats):** A non-overlapping range defining acceptable values for beta attenuation.

- **`linear_depolarisation_ratio_range` (list of floats):** A non-overlapping range specifying acceptable values for linear depolarization ratio.

- **`color` (string):** Color code used for visualization, e.g., "#a9cac4."

_Note: It's recommended to use all lower integers for `class_id` initially, but the order does not need to be sorted in the JSON._

Feel free to add more classes to the `classes` array based on your specific classification needs. Keep simply in mind that overlapping ranges are not managed here. Adjusting these parameters lets you fine-tune the classification process according to your specific requirements.

## Contributors

- Geza Soldati: Master's student at EPFL Lausanne (switzerland), geza.soldati@epfl.ch

## License

This project is licensed under the 3-Clause BSD License.


## Acknowledgments
I would like to thank Prof. Julia Schmale and the whole EERL laboratory for providing the opportunity to work with exciting instruments and datasets from their campaigns in extreme environments. I particularly want to thank Benjamin Heutte who closely supervised my work throughout the semester.

##
For any questions or inquiries, please feel free to contact the project's contributor mentioned above.
