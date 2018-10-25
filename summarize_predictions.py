import os
from glob import glob
import pandas as pd
from dask import delayed
from sklearn.metrics import f1_score
import numpy as np
from functools import reduce
from padar_parallel.for_loop import ForLoop


def pa_to_activity_group(prediction_set):
    mapping = prediction_set.groupby('ACTIVITY').apply(
        lambda chunk: chunk['ACTIVITY_GROUP'].values[0]).to_dict()
    prediction_set.loc[:, 'ACTIVITY_GROUP_PREDICTION'] = prediction_set['ACTIVITY_PREDICTION'].map(
        mapping)
    labels = list(set(mapping.values()))
    return prediction_set, labels


def pa_metric(prediction_set):
    prediction_set, labels = pa_to_activity_group(prediction_set)
    report = {}
    average_f1_inter = f1_score(
        y_true=prediction_set['ACTIVITY_GROUP'], y_pred=prediction_set['ACTIVITY_GROUP_PREDICTION'], average='macro')
    f1s_inter = f1_score(
        y_true=prediction_set['ACTIVITY_GROUP'], y_pred=prediction_set['ACTIVITY_GROUP_PREDICTION'], labels=labels, average=None)
    report['ACTIVITY_AVERAGE'] = [f1_score(
        y_true=prediction_set['ACTIVITY'], y_pred=prediction_set['ACTIVITY_PREDICTION'], average='macro')]
    report['ACTIVITY_GROUP_AVERAGE'] = [average_f1_inter]
    for label, f1_inter in zip(labels, f1s_inter):
        report[label.upper() + "_GROUP"] = [f1_inter]
        if label == 'Biking' or label == 'Lying' or label == 'Running':
            continue
        prediction_set_inner = prediction_set[prediction_set['ACTIVITY_GROUP'] == label]
        prediction_set_inner = prediction_set_inner[prediction_set_inner['ACTIVITY_GROUP_PREDICTION'] == label]
        report[label.upper() + "_IN_GROUP"] = [f1_score(
            y_true=prediction_set_inner['ACTIVITY'], y_pred=prediction_set_inner['ACTIVITY_PREDICTION'], average='macro')]
    report = pd.DataFrame(data=report)
    report.insert(2, 'ACTIVITY_IN_GROUP_AVERAGE', np.mean(
        report.select(lambda x: 'IN_GROUP' in x, axis=1).values, axis=1))
    report = report.reset_index(drop=True)
    return(report)


def posture_metric(prediction_set):
    report = {}
    y_true = prediction_set['POSTURE']
    y_pred = prediction_set["POSTURE_PREDICTION"]
    labels = np.union1d(np.unique(y_true), np.unique(y_pred)).tolist()
    f1_average = f1_score(y_true, y_pred, average='macro')
    f1_classes = f1_score(y_true, y_pred, labels=labels, average=None)
    report['POSTURE_AVERAGE'] = [f1_average]
    for label, f1_class in zip(labels, f1_classes):
        report[label.upper() + '_POSTURE'] = [f1_class]
    report = pd.DataFrame(data=report)
    return(report)


def summarize_prediction_set_file(prediction_set_file, targets):
    prediction_set = delayed(pd.read_csv)(
        prediction_set_file, parse_dates=[0, 1], infer_datetime_format=True)
    return summarize_prediction_set(prediction_set, targets)


@delayed
def summarize_prediction_set(prediction_set, targets):
    metrics = []
    placements = prediction_set['SENSOR_PLACEMENT'].values[0]
    feature_type = prediction_set['FEATURE_TYPE'].values[0]
    num_of_sensors = len(placements.split('_'))
    for target in targets:
        if target == 'POSTURE':
            report = posture_metric(prediction_set)
        elif target == 'ACTIVITY':
            report = pa_metric(prediction_set)
        report.insert(0, 'SENSOR_PLACEMENT', placements)
        report.insert(1, 'NUM_OF_SENSORS', num_of_sensors)
        report.insert(2, 'FEATURE_TYPE', feature_type)
        metrics.append(report)
    print('processed ' + placements + ' ' + feature_type)
    return reduce(pd.merge, metrics)


def summarize_predictions(prediction_set_folder):
    prediction_set_files = glob(os.path.join(
        prediction_set_folder, '*.prediction.csv'))
    targets = ['POSTURE', 'ACTIVITY']
    experiment = ForLoop(
        prediction_set_files, summarize_prediction_set_file, merge_func=delayed(lambda x, **kwargs: pd.concat(x, axis=0)), targets=targets)
    experiment.compute(scheulder='sync')
    result = experiment.get_result()
    # sort result
    result = result.sort_values(by=['NUM_OF_SENSORS', 'FEATURE_TYPE'])
    result.to_csv(os.path.join(prediction_set_folder,
                               'summary.csv'), index=False, float_format='%.6f')


if __name__ == '__main__':
    dataset_folder = 'D:/data/spades_lab/'
    prediction_set_folder = os.path.join(
        dataset_folder, 'DerivedCrossParticipants', 'location_matters', 'prediction_sets')
    summarize_predictions(prediction_set_folder)
