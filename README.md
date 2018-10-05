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

3. Download original dataset and unzip to a directory

4. Reproduce the feature set

```python
python prepare_feature_set.py your_dataset_path sampling_rate=80
python prepare_class_set.py your_dataset_path
```

5. Reproduce validation results

6. Reproduce publication figures and tables
