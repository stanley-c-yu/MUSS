import os
from glob import glob
import pandas as pd
from dask import delayed
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from functools import reduce
from padar_parallel.for_loop import ForLoop
from helper.utils import generate_run_folder, strip_path
from clize import run
import logging


def pa_to_activity_group(prediction_set, target):
    mapping = prediction_set.groupby(target).apply(
        lambda chunk: chunk['MUSS_6_ACTIVITY_GROUPS'].values[0]).to_dict()
    prediction_set.loc[:, 'MUSS_6_ACTIVITY_GROUPS_PREDICTION'] = prediction_set[
        target + '_PREDICTION'].map(mapping)
    labels = list(set(mapping.values()))
    return prediction_set, labels


def get_pa_labels(input_folder, targets):
    filepath = os.path.join(input_folder, 'MetaCrossParticipants',
                            'muss_class_labels.csv')
    label_mapping = pd.read_csv(filepath)
    labels = label_mapping[targets].unique().tolist()
    labels.remove('Unknown')
    labels.remove('Transition')
    return labels


def pa_confusion_matrix(prediction_set, labels, target):
    y_true = prediction_set[target]
    y_pred = prediction_set[target + '_PREDICTION']
    conf_mat = confusion_matrix(y_true, y_pred, labels)
    conf_df = pd.DataFrame(conf_mat, columns=labels, index=labels)
    return conf_df


def pa_metric(prediction_set, target):
    prediction_set, labels = pa_to_activity_group(prediction_set, target)
    report = {}
    average_f1_inter = f1_score(
        y_true=prediction_set['MUSS_6_ACTIVITY_GROUPS'],
        y_pred=prediction_set['MUSS_6_ACTIVITY_GROUPS_PREDICTION'],
        average='macro')
    f1s_inter = f1_score(
        y_true=prediction_set['MUSS_6_ACTIVITY_GROUPS'],
        y_pred=prediction_set['MUSS_6_ACTIVITY_GROUPS_PREDICTION'],
        labels=labels,
        average=None)
    report[target + '_AVERAGE'] = [
        f1_score(
            y_true=prediction_set[target],
            y_pred=prediction_set[target + '_PREDICTION'],
            average='macro')
    ]
    report['MUSS_6_ACTIVITY_GROUPS_INTER_AVERAGE'] = [average_f1_inter]
    for label, f1_inter in zip(labels, f1s_inter):
        report[label.upper() + "_GROUP"] = [f1_inter]
        if label == 'Biking' or label == 'Lying' or label == 'Running':
            continue
        prediction_set_inner = prediction_set[prediction_set['MUSS_6_ACTIVITY_GROUPS']
                                              == label]
        prediction_set_inner = prediction_set_inner[
            prediction_set_inner['MUSS_6_ACTIVITY_GROUPS_PREDICTION'] == label]
        report[label.upper() + "_IN_GROUP"] = [
            f1_score(
                y_true=prediction_set_inner[target],
                y_pred=prediction_set_inner[target + '_PREDICTION'],
                average='macro')
        ]
    report = pd.DataFrame(data=report)
    report.insert(
        2, 'MUSS_6_ACTIVITY_GROUPS_INNER_AVERAGE',
        np.mean(
            report.loc[:,list(filter(lambda x: 'IN_GROUP' in x, report.columns))].values, axis=1))
    report = report.reset_index(drop=True)
    return (report)


def posture_metric(prediction_set):
    report = {}
    y_true = prediction_set['MUSS_3_POSTURES']
    y_pred = prediction_set["MUSS_3_POSTURES_PREDICTION"]
    labels = np.union1d(np.unique(y_true), np.unique(y_pred)).tolist()
    f1_average = f1_score(y_true, y_pred, average='macro')
    f1_classes = f1_score(y_true, y_pred, labels=labels, average=None)
    report['MUSS_3_POSTURES_AVERAGE'] = [f1_average]
    for label, f1_class in zip(labels, f1_classes):
        report[label.upper() + '_POSTURE'] = [f1_class]
    report = pd.DataFrame(data=report)
    return (report)


def compute_metrics_for_single_file(prediction_set_file, targets, model_type,
                                    dataset_folder):
    prediction_set = delayed(pd.read_csv)(
        prediction_set_file, parse_dates=[0, 1], infer_datetime_format=True)
    result = _compute_metrics(dataset_folder, prediction_set, targets, model_type, prediction_set_file)

    return result


def save_confusion_matrix(prediction_set_file, model_type, conf):
    output_filepath = prediction_set_file.replace(
        'predictions', 'confusion_matrices').replace('.' + model_type + '_prediction.csv',
                                                     '.' + model_type + '_confusion_matrix.csv')
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    conf.to_csv(output_filepath, index=True, float_format='%.3f')


@delayed
def _compute_metrics(input_folder, prediction_set, targets, model_type, prediction_set_file):
    metrics = []
    placements = prediction_set['SENSOR_PLACEMENT'].values[0]
    feature_type = prediction_set['FEATURE_TYPE'].values[0]
    num_of_sensors = len(placements.split('_'))
    for target in targets:
        if target == 'MUSS_3_POSTURES':
            report = posture_metric(prediction_set)
        else:
            pa_labels = get_pa_labels(input_folder, target)
            report = pa_metric(prediction_set, target)
            conf_df = pa_confusion_matrix(prediction_set, pa_labels, target)
            save_confusion_matrix(prediction_set_file, model_type, conf_df)
            print('saved confusion matrix: ' + placements + ' ' + feature_type)
        report.insert(0, 'SENSOR_PLACEMENT', placements)
        report.insert(1, 'NUM_OF_SENSORS', num_of_sensors)
        report.insert(2, 'FEATURE_TYPE', feature_type)
        metrics.append(report)
    print('processed ' + placements + ' ' + feature_type)
    return reduce(pd.merge, metrics)


def main(input_folder, *, output_folder=None, debug=False, scheduler='processes', profiling=True, force=True, target=None,model_type='svm', include_nonwear=False):
    """Compute metrics for the validation predictions.

    :param input_folder: Folder path of input raw dataset
    :param output_folder: Auto path if None
    :param debug: Use this flag to output results to 'debug_run' folder
    :param scheduler: 'processes': Use multi-core processing;
                      'threads': Use python threads (not-in-parallel);
                      'sync': Use a single thread in sequential order
    :param profiling: use profiling or not
    """
    if output_folder is None:
        output_folder = generate_run_folder(input_folder, debug=debug)

    metric_file = os.path.join(output_folder, 'muss.' + model_type + '_metrics.csv')
    cm_folder = os.path.join(output_folder, 'confusion_matrices')

    if not force and os.path.exists(metric_file) and os.path.exists(cm_folder):
        logging.info('Metric and confusion matrices exist, skip regenerating them...')
        return metric_file, cm_folder

    suffix = '_with_nonwear' if include_nonwear else ''
    if target is None:
        prediction_set_folder = os.path.join(output_folder, 'predictions' + suffix)
    else:
        prediction_set_folder = os.path.join(output_folder, target + '_predictions' + suffix)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(prediction_set_folder, exist_ok=True)

    prediction_set_files = glob(
        os.path.join(prediction_set_folder, '*.' + model_type + '_prediction.csv'))
    if target is None:
        targets = ['MUSS_22_ACTIVITIES', 'MUSS_3_POSTURES']
    else:
        targets = target.split(',')
    experiment = ForLoop(
        prediction_set_files,
        compute_metrics_for_single_file,
        merge_func=delayed(lambda x, **kwargs: pd.concat(x, axis=0)),
        targets=targets,
        model_type=model_type,
        dataset_folder=input_folder)
    experiment.compute(scheulder=scheduler, profiling=profiling)
    result = experiment.get_result()
    # sort result
    result = result.sort_values(by=['NUM_OF_SENSORS', 'FEATURE_TYPE'])
    result.to_csv(
        metric_file,
        index=False,
        float_format='%.6f')
    return metric_file, cm_folder 


if __name__ == '__main__':
    run(main)
