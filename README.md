__Under development__

# Multi- vs. single-site accelerometer sensing for posture and activity recognition using machine learning

_source codes and data_

## Corresponding author

<div class="github-card" data-github="qutang" data-width="400" data-height="150" data-theme="default"></div>
<script src="//cdn.jsdelivr.net/github-cards/latest/widget.js"></script>

## Citation

In-submission.

## Replicate results using jupyter notebook in your local runtime

We provide a step by step guide in a jupyter notebook to reproduce the results and see the intermediate results.

1. Make sure you follow the __dependencies__, __step 1__ and __step 2__ in [Run from a terminal in your local runtime](#run-from-a-terminal-in-your-local-runtime).

2. Install jupyter notebook

    ```bash
    > pip install jupyter
    ```

3. Start jupyter notebook from the root folder of the code repository

    ```bash
    > jupyter notebook
    ```

4. Open `http://localhost:8888` in a browser and open `muss_guide.ipynb` and follow the guidance in the notebook.

## Replicate results from a terminal in your local runtime

### Dependencies

1. Python 3.6.5 or above
2. `pipenv` package. Install using `pip install pipenv`
3. `git` (optional)
4. Recommend at least 8-core workstation, otherwise computation will be really slow
5. [`graphviz`](https://www.graphviz.org/download/) (optional, required to generate workflow diagram pdf)

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

---

> Now we will start reproducing the results using the dataset. All results will be in the output folder: 
```
DerivedCrossParticipants/product_run/
``` 
inside the dataset folder. If `--debug` flag is turned on for the following scripts, it will use `debug_run` instead of `product_run` for the folder name of all results.

### Generate class labels from annotations

```bash
> pipenv run python prepare_class_set.py ./muss_data/ --scheduler=processes
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

### Generate features from raw sensory data

```bash
> pipenv run python prepare_class_set.py ./muss_data/ --sampling-rate=80 --scheduler=processes
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

### Generate validation datasets for each experiment

Because features and class labels are stored in a `long-form` for the entire dataset (for all placements, participants), we want to divide and combine different subsets together to form validation datasets for different experiments.

For example, we want to extract data of "motion" features and "posture" and "activity" class labels from dominant wrist (DW) and dominant ankle (DA) to form a dataset to evaluate the performance of a model with DW + DA and "motion" features on posture and activity recognition.

```bash
> pipenv run python prepare_validation_set.py ./muss_data/ --scheduler=processes
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

### Run LOSO validation on each validation dataset

```bash
> pipenv run python run_validation_experiments.py ./muss_data/ --scheduler=processes
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

### Compute metrics across all prediction results

You may find the descriptions of all metrics in the manuscript.

```bash
> pipenv run python compute_metrics.py ./muss_data/ --scheduler=processes
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

### Generate figures and tables used in the manuscript

```bash
> pipenv run python publication_figures.py ./muss_data/
```

You may run,

```bash
> pipenv run python publication_figures.py --help
```

for inline help information.

You will see a `figures_and_tables` sub-folder in the output folder.
* **`figures_and_tables/*.*`**: the figure and table files in various file format (e.g., png, pdf, xlsx, csv). The filenames are corresponding to the numbering in the manuscript.

## Run model on your own data

### Train and save models using generated dataset

1. Make sure you have completed [generated validation datasets](#generate-validation-datasets-for-each-experiment).

2. Use following script to train a model using one of the entire validation dataset.

    For example, if you need to train DW + DA models with 'MO' (Motion + Orientation) feature sets to classify 3-class postures and 22-classes activities, you may use the following script.

    ```bash
    > pipenv run python train_and_save_model.py ./muss_data/ --targets=ACTIVITY,POSTURE --feature-set=MO --sites=DW,DA
    ```

    The saved model will be `pickle` files with names `DW_DA.MO.posture_model.pkl` and `DW_DA.MO.activity_model.pkl` in the `models` subfolder of the output folder.

### Run a saved model on your own dataset

1. Prepare your own dataset according to the structure of the `muss_data`.

2. Run according to [Replicate results from a terminal in your local runtime](#replicate-results-from-a-terminal-in-your-local-runtime) until you complete step [generated validation datasets](#generate-validation-datasets-for-each-experiment).

3. Run a saved model on the generate dataset.

    For example, if you want to run the model generated in step [Train and save models using generated dataset](#train-and-save-models-using-generated-dataset) on a newly generated dataset (`DW_DA.MO.dataset.csv`) in current input folder.

    First, copy the full path of the generated posture model file. Here we denote it as `POSTURE_MODEL_PATH`.

    Then, run the following script,

    ```bash
    > pipenv run python run_saved_model.py --dataset-file=DW_DA.MO.dataset.csv --model-file=POSTURE_MODEL_PATH
    ```

    And you will find the generated prediction file `DW_DA.MO.posture_prediction.csv` in the `tests` subfolder of the output folder of your NEW dataset.

    __Note that `--model-file` should be the full path of a model file, while `--dataset-file` should be the filename of a dataset file to test in current dataset.__