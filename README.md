__Under development__

# Multi- vs. single-site accelerometer sensing for posture and activity recognition using machine learning

_source codes and data_

## Corresponding author

Qu Tang. Contact me via github issues. https://github.com/qutang/MUSS

## Citation

In-submission.

## Run using Colab notebook

We provide a step by step guide on Google Colab to reproduce the results and see the intermediate results. Due to limited computing resources, if using Colab's hosted runtime environment, it takes very long time (2-3h) to finish the entire process. If you do not want to wait, you may set up your own local runtime according to Colab's guidance. The source codes are optimized to use parallel computing (based on `Dask` package) on multi-core machines. It is recommended to use at least 8-core (more the faster) workstation for this project.

## Run from a terminal in your local runtime

### Dependencies

1. Python 3.6.5 or above
2. `pipenv` package. Install using `pip install pipenv`
3. `git` (optional)

### Download source codes and data

1. Download source code

    If you have `git` installed, run following command in terminal,

    ```bash
    > git clone https://github.com/qutang/MUSS.git
    > cd MUSS
    ```

    If you do not have `git` installed, go to https://github.com/qutang/MUSS/archive/master.zip and use your browser to download the code repository. Then unzip it. Open a terminal and navigate into the unzipped folder.

    Make sure you are in the root folder of the code repository from now on.

2. Install dependencies

    ```bash
    > pipenv install
    ```

    If you are updating from a previous version, then run,

    ```bash
    > pipenv update
    ```

3. Download and unzip dataset

    If you have `wget` and `tar` installed, run,
    ```bash
    > wget https://github.com/qutang/MUSS/releases/download/data/muss_data.tar.gz
    > tar -xzf muss_data.tar.gz -C ./
    ```

    Make sure the dataset folder is in the root folder of the code repository.

### Reproduce results

Now we will start reproducing the results using the dataset. All results will be in the output folder: 
```
DerivedCrossParticipants/[CURRENT_DATE]/product_run/
``` 
inside the dataset folder. If `--debug` flag is turned on for the following scripts, it will use `debug_run` instead of `product_run` for the folder name of all results.

1. Generate class labels from annotations

    ```bash
    > pipenv run python prepare_class_set.py ./muss_data/ scheduler="processes"
    ```

    You may run,
    
    ```bash
    > pipenv run python prepare_class_set.py --help
    ```
    for inline help information.

    You will see four output files in the output folder.
    * `classset_computation_workflow.pdf`: the computational graph
    * `classset_computation_profiling.html`: the computational profiling (memory, cpu, and disk) visualization
    * **`muss.class.csv`**: the class labels by segmentation window (12.8s) for the whole dataset.
    * `muss.classmap.csv`: intermediate file used to parse class labels.

2. Generate features from raw sensory data

    ```bash
    > pipenv run python prepare_class_set.py ./muss_data/ sampling-rate=80 scheduler="processes"
    ```

    You may run,
    
    ```bash
    > pipenv run python prepare_class_set.py --help
    ```

    for inline help information.

    You will see three output files in the output folder.
    * `feature_computation_workflow.pdf`: the computational graph
    * `feature_computation_profiling.html`: the computational profiling (memory, cpu, and disk) visualization
    * **`muss.feature.csv`**: the feature values by segmentation window (12.8s) for the whole dataset.

3. Generate validation datasets for each experiment

    Because features and class labels are stored in a `long-form` for the entire dataset (for all placements, participants), we want to divide and combine different subsets together to form validation datasets for different experiments.

    For example, we want to extract data of "motion" features and "posture" and "activity" class labels from dominant wrist (DW) and dominant ankle (DA) to form a dataset to evaluate the performance of a model with DW + DA and "motion" features on posture and activity recognition.

    ```bash
    > pipenv run python prepare_validation_set.py ./muss_data/ scheduler="processes"
    ```

    You may run,
    
    ```bash
    > pipenv run python prepare_validation_set.py --help
    ```

    for inline help information.

    You will see two output files and a `datasets` sub-folder in the output folder.
    * `dataset_computation_workflow.pdf`: the computational graph
    * `dataset_computation_profiling.html`: the computational profiling (memory, cpu, and disk) visualization
    * **`datasets/*.dataset.csv`**: the validation dataset files for different sensor combinations and feature sets.

4. Run LOSO validation on each validation dataset

    ```bash
    > pipenv run python run_validation_experiments.py ./muss_data/ scheduler="processes" 
    ```

    You may run,
    
    ```bash
    > pipenv run python run_validation_experiments.py --help
    ```

    for inline help information.

    You will see two output files and a `predictions` sub-folder in the output folder.
    * `dataset_computation_workflow.pdf`: the computational graph
    * `dataset_computation_profiling.html`: the computational profiling (memory, cpu, and disk) visualization
    * **`predictions/*.prediction.csv`**: the prediction results per window for each validation dataset.

5. Compute metrics across all prediction results

    You may find the descriptions of all metrics in the manuscript.
   
    ```bash
    > pipenv run python compute_metrics.py ./muss_data/ scheduler="processes"
    ```

    You may run,
    
    ```bash
    > pipenv run python run_validation_experiments.py --help
    ```

    for inline help information.

    You will see three output files and a `confusion_matrices` sub-folder in the output folder.
    * `dataset_computation_workflow.pdf`: the computational graph
    * `dataset_computation_profiling.html`: the computational profiling (memory, cpu, and disk) visualization
    * **`muss.metrics.csv`**: the metrics summary across all validation datasets.
    * **`confusion_matrices/*.confusion_matrix.csv`**: the confusion matrix of activity recognition (not posture recognition) for each validation dataset.

6. Generate figures and tables used in the manuscript

    ```bash
    > pipenv run python publication_figures.py ./muss_data/
    ```

    You may run,
    
    ```bash
    > pipenv run python run_validation_experiments.py --help
    ```

    for inline help information.

    You will see a `figures_and_tables` sub-folder in the output folder.
    * **`figures_and_tables/*.*`**: the figure and table files in various file format (e.g., png, pdf, xlsx, csv). The filenames are corresponding to the numbering in the manuscript.