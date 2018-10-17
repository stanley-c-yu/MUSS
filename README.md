__Under development__

# Intro

The source code and feature files to reproduce the results and publication figures for "Location Matters: Impact of Single vs. Multi-site Motion Sensing and Dominant vs. Non-dominant Wrist When Using a Machine Learning Algorithm to Measure Postures and Physical Activities"

# Citation

TBA

# Prerequistes

1. Python 3.6.5
2. Install `pipenv` package `pip install pipenv`

# Step

1. Open a terminal and navigate to the root path `your/path/to/location_matters/`

2. Install all dependencies and activate the shell to run the scripts

    ```python
    pipenv install
    pipenv shell
    ```

    If you are updating from a previous existing codes, run 

    ```python
    pipenv update
    ```

3. Download original dataset and unzip to a directory `your/path/to/dataset`

4. Reproduce the feature set and prepare the class labels

    ```bash
    python prepare_feature_set.py your/path/to/dataset your/path/to/output/results sampling-rate=80 scheduler=processes
    python prepare_class_set.py your/path/to/dataset your/path/to/output/results sampling-rate=80 scheduler=processes
    ```

    Run following command for help information.
    ```bash
    python prepare_feature_set --help
    python prepare_class_set --help
    ```

5. Reproduce validation results

    The first step is to prepare validation sets (for LOSO validation) for different sensor combinations
    ```bash
    python prepare_validation_set.py your/path/to/output/results
    ```
    The script will generate files for each sensor combinations and store them separately in `your/path/to/output/results/validation_sets/` folder.


    The second step is to run LOSO validation on each of the validation sets to generate the classification results.

    TBA.

6. Reproduce publication figures and tables
