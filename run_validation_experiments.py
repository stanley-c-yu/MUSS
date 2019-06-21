import pandas as pd
from helper.svm_model import loso_validation
from helper import utils
from dask import delayed
from itertools import product
from padar_parallel.for_loop import ForLoop
from glob import glob
import os
from helper.utils import generate_run_folder
from clize import run
import logging


def run_all_experiments(dataset_folder, scheduler='processes', profiling=True):
    output_folder = dataset_folder.replace('datasets', 'predictions')
    os.makedirs(output_folder, exist_ok=True)
    validation_files = glob(os.path.join(dataset_folder, '*.dataset.csv'))
    experiments = ForLoop(validation_files, run_single_experiment)
    experiments.compute(scheduler=scheduler, profiling=profiling)
    if profiling:
        try:
            experiments.show_workflow(
                os.path.join(
                    os.path.dirname(output_folder),
                    'validation_experiment_workflow.pdf'))
        except Exception as e:
            print(e)
        print('skip generating workflow pdf')
        experiments.show_profiling(
            os.path.join(
                os.path.dirname(output_folder),
                'validation_experiment_profiling.html'))
    prediction_sets = experiments.get_result()
    # save_prediction_sets(prediction_sets)


@delayed
def exclude_unknown_and_transition(validation_set, target):
    exclude_criterion = (validation_set[target] != 'Unknown') & (
        validation_set[target] != 'Transition')
    validation_set = validation_set.loc[exclude_criterion, :]
    return validation_set


def run_single_experiment(validation_set_file, target=None):
    validation_set = delayed(pd.read_csv)(
        validation_set_file, parse_dates=[0, 1], infer_datetime_format=True)
    if target is not None:
        targets = [target]
    else:
        targets = ['POSTURE', 'ACTIVITY']
    predictions = {}
    for target in targets:
        validation_set = exclude_unknown_and_transition(
        validation_set, target=target)
        prediction = run_loso(validation_set, target)
        predictions[target + '_PREDICTION'] = prediction
    prediction_set = append_prediction(validation_set, predictions)
    return save_prediction_set(prediction_set, validation_set_file)


@delayed
def save_prediction_set(prediction_set, validation_set_file):
    placements = prediction_set['SENSOR_PLACEMENT'].values[0]
    print('Saving prediction set for: ' + placements)
    output_filepath = validation_set_file.replace(
        'datasets', 'predictions').replace('dataset.csv', 'prediction.csv')
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    prediction_set.to_csv(output_filepath, index=False, float_format='%.9f')
    return prediction_set


def save_prediction_sets(prediction_sets):
    for prediction_set in prediction_sets:
        placements = prediction_set['DATA']['SENSOR_PLACEMENT'].values[0]
        print('Saving prediction set for: ' + placements)
        output_filepath = prediction_set['FILE'].replace(
            'validation_sets', 'prediction_sets').replace(
                'dataset.csv', 'prediction.csv')
        prediction_set['DATA'].to_csv(
            output_filepath, index=False, float_format='%.9f')


@delayed
def append_prediction(validation_set, predictions):
    validation_set = validation_set.assign(**predictions)
    return validation_set


@delayed
def run_loso(validation_set, target):
    index_cols = [
        "START_TIME", "STOP_TIME", "PID", "SID", "SENSOR_PLACEMENT",
        "FEATURE_TYPE", "ANNOTATOR", "ANNOTATION_LABELS", "ACTIVITY",
        "POSTURE", "ACTIVITY_GROUP", "MDCAS", "THIRTEEN_ACTIVITIES", "CLASSIC_SEVEN_ACTIVITIES",
        "SEDENTARY_AMBULATION_CYCLING", 'ACTIVITY_ABBR'
    ]
    placements = validation_set['SENSOR_PLACEMENT'].values[0]
    feature_type = validation_set['FEATURE_TYPE'].values[0]
    y = validation_set[target].values
    indexed_validation_set = validation_set.set_index(index_cols)
    X = indexed_validation_set.values
    groups = validation_set['PID'].values
    y_pred, metric = loso_validation(X, y, groups=groups)
    print(placements + "'F1-score, using " + feature_type + " features for " +
          target + ' is: ' + str(metric))
    return y_pred


def main(input_folder, *, output_folder=None, debug=False, scheduler='processes', profiling=True, force=True, sites=None, feature_set=None, target=None):
    """Run validation experiments.

    :param input_folder: Folder path of input raw dataset
    :param output_folder: auto path if None
    :param debug: Use this flag to output results to 'debug_run' folder
    :param scheduler: 'processes': Use multi-core processing;
                      'threads': Use python threads (not-in-parallel);
                      'sync': Use a single thread in sequential order
    :param profiling: use profiling or not
    :param sites: if `None`, the function will run as for result reproduction. Otherwise, it will run for a single selected validation dataset. Should be comma separated string (e.g., "DW,DA").
    :param feature_set: if `None`, the function will run as for result reproduction. Otherwise, it will run for a single selected validation dataset. Should be either "MO", "O" or "M".
    :param target: if `None`, the function will run as for result reproduction. Otherwise, it will run for a single selected validation dataset. Should be one of the class column names.

    """
    if output_folder is None:
        output_folder = generate_run_folder(
            input_folder, debug=debug)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    prediction_folder = os.path.join(output_folder, 'predictions')

    if not force and os.path.exists(prediction_folder):
        logging.info("Prediction folder exists, skip regenerating it...")
        return prediction_folder
    
    dataset_folder = os.path.join(output_folder, 'datasets')
    if sites is None or feature_set is None or target is None:
        run_all_experiments(dataset_folder, scheduler=scheduler, profiling=profiling)
    else:
        sites = sites.split(',')
        sites.sort()
        sites = '_'.join(sites)
        validation_file = os.path.join(
            dataset_folder, sites + '.' + feature_set + '.dataset.csv')
        print(validation_file)
        experiments = ForLoop(
            [validation_file], run_single_experiment, target=target)
        experiments.compute(scheduler='sync')
    return prediction_folder


if __name__ == '__main__':
    # main('D:/data/muss_data/', debug=True, scheduler='processes',
    #      sites='DW', feature_set='MO', target='MDCAS')
    run(main)
