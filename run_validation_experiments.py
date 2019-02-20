import pandas as pd
from helper.svm_model import loso_validation
from dask import delayed
from itertools import product
from padar_parallel.for_loop import ForLoop
from glob import glob
import os
from helper.utils import generate_run_folder


def run_all_experiments(dataset_folder, scheduler='processes'):
    output_folder = dataset_folder.replace('datasets', 'predictions')
    os.makedirs(output_folder, exist_ok=True)
    validation_files = glob(os.path.join(dataset_folder, '*.dataset.csv'))
    experiments = ForLoop(validation_files, run_single_experiment)
    experiments.show_workflow(
        os.path.join(
            os.path.dirname(output_folder), 'logs',
            'validation_experiment_workflow.pdf'))
    experiments.compute(scheduler=scheduler)
    experiments.show_profiling(
        os.path.join(
            os.path.dirname(output_folder), 'logs',
            'validation_experiment_profiling.html'))
    prediction_sets = experiments.get_result()
    # save_prediction_sets(prediction_sets)


def run_single_experiment(validation_set_file):
    validation_set = delayed(pd.read_csv)(
        validation_set_file, parse_dates=[0, 1], infer_datetime_format=True)
    targets = ['POSTURE', 'ACTIVITY']
    predictions = {}
    for target in targets:
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
        "POSTURE", "ACTIVITY_GROUP", "SEDENTARY_AMBULATION_CYCLING",
        'ACTIVITY_ABBR'
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


if __name__ == '__main__':
    input_folder = 'D:/data/spades_lab/'
    input_folder = 'D:/data/mini_mhealth_dataset_cleaned/'
    output_folder = generate_run_folder(input_folder, debug=False)
    dataset_folder = os.path.join(output_folder, 'datasets')
    os.makedirs(os.path.join(output_folder, 'logs'), exist_ok=True)
    run_all_experiments(dataset_folder, scheduler='processes')
