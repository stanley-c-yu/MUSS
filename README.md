# Multi-site sensing for activity recognition using accelerometers

This repo hosts the source codes and dataset for "Posture and Physical Activity Detection: Impact of Number of Sensors and Feature Type" (doi:10.1249/MSS.0000000000002306).

## Citation

If you have used the codes or referred to the manuscript in your publication, please kindly cite the following paper.

> Tang, Q., John, D., Thapa-Chhetry, B., Arguello, D.J. and Intille, S., 2020. Posture and Physical Activity Detection: Impact of Number of Sensors and Feature Type. Medicine & Science in Sports & Exercise. Preprint.

## Dependencies

1. Python >= 3.7
2. `poetry` dependency management for python. Install using `pip install poetry`.
3. `git`
5. [`graphviz`](https://www.graphviz.org/download/) (optional, required to generate workflow diagram pdf)

## Get started

### Install dependencies

At the root of the project folder, run

```bash
> poetry install
```

## Reproduce publication results

Run with multi-core processing and memory on a new session folder. Overwrite data or results if they are found to exist.

```bash
>> pipenv run reproduce --parallel --force-fresh-data=True
```

You may find intermediate and publication results in `./muss_data/DerivedCrossParticipants` in a folder prefixed with `product_run`.

### Sample reproduction results

Check [here](https://github.com/qutang/MUSS/releases/latest/download/sample_reproduction_results.tar.gz) for a sample reproduction results.

### Get help 

Run above command at the root of the repository to see the usage of the reproduction script.

```bash
>> poetry run python reproduce.py --help
```

The reproduct script will do the following,

1. Download and unzip the raw dataset
2. Compute features and convert annotations to class labels
3. Run LOSO cross validations on datasets of different combinations of sensor placements
4. Compute metrics from the outputs of the LOSO cross validations
5. Generate publication tables and graphs 

## Train and save a model using muss dataset

Make sure you run the `reproduce` script at first.

You may want to train and save an activity recognition model using the dataset shipped with this repo. To do it, at the root of the project folder, run,

```bash
>> poetry run python train_and_save_model.py
```

By default, it will train two dual-sensor models (DW + DA) using motion + orientation feature set, each for postures and daily activities.

### Get help

```bash
>> poetry run python train_and_save_model.py --help
```

You may config the type of feature set, the target class labels and the sensor placements used.