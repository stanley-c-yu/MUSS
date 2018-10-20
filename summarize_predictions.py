import os
from glob import glob
import pandas as pd
from dask import delayed
from sklearn.metrics import f1_score
import numpy as np
from functools import reduce
from padar_parallel.for_loop import ForLoop


def summarize_prediction_set_file(prediction_set_file, targets):
    prediction_set = delayed(pd.read_csv)(
        prediction_set_file, parse_dates=[0, 1], infer_datetime_format=True)
    return summarize_prediction_set(prediction_set, targets)


def performance_report(x):
    report = {}
    average_f1_inter = f1_score(
        y_true=x['merged_activity'], y_pred=x['merged_predicted_activity'], labels=labels, average='macro')
    f1s_inter = f1_score(
        y_true=x['merged_activity'], y_pred=x['merged_predicted_activity'], labels=labels, average=None)
    report['inter_average_f1'] = [average_f1_inter]
    report['average_f1'] = f1_score(
        y_true=x['activity'], y_pred=x['predicted_activity'], average='macro')
    i = 0
    for label in labels:
        report[label + "_inter_f1_score"] = [f1s_inter[i]]
        i = i + 1
    for label in labels:
        if label == 'B' or label == 'L' or label == 'R':
            continue
        x_inner = x[x['merged_activity'] == label]
        x_inner = x_inner[x_inner['merged_predicted_activity'] == label]
        report[label + "_inner_f1_score"] = f1_score(
            y_true=x_inner['activity'], y_pred=x_inner['predicted_activity'], average='macro')
    report = pd.DataFrame(data=report)
    return(report)


report_df = prediction_df3.groupby(
    ['location', 'second_location', 'feature']).apply(performance_report)
report_df['inner_average_f1'] = np.mean(report_df.select(
    lambda x: 'inner' in x, axis=1).values, axis=1)
report_df = report_df.reset_index(drop=False)


@delayed
def summarize_prediction_set(prediction_set, targets):

    metrics = []
    for target in targets:
        y_true = prediction_set[target]
        y_pred = prediction_set[target + "_PREDICTION"]
        placements = prediction_set['SENSOR_PLACEMENT'].values[0]
        feature_type = prediction_set['FEATURE_TYPE'].values[0]
        num_of_sensors = len(placements.split('_'))
        labels = np.union1d(np.unique(y_true), np.unique(y_pred)).tolist()
        f1_average = f1_score(y_true, y_pred, average='macro')
        f1_classes = f1_score(y_true, y_pred, average=None)
        result = pd.DataFrame(data=[[placements, num_of_sensors, feature_type, f1_average] + f1_classes.tolist()], columns=[
                              'SENSOR_PLACEMENT', 'NUM_OF_SENSORS', 'FEATURE_TYPE', target + '_AVERAGE'] + labels, index=[0])
        metrics.append(result)
    print('processed ' + placements + ' ' + feature_type)
    return reduce(pd.merge, metrics)


def summarize_predictions(prediction_set_folder):
    prediction_set_files = glob(os.path.join(
        prediction_set_folder, '*.prediction.csv'))
    targets = ['POSTURE', 'ACTIVITY']
    experiment = ForLoop(
        prediction_set_files, summarize_prediction_set_file, merge_func=delayed(lambda x, **kwargs: pd.concat(x, axis=0)), targets=targets)
    experiment.compute(scheulder='processes')
    result = experiment.get_result()
    result.to_csv(os.path.join(prediction_set_folder,
                               'summary.csv'), index=False, float_format='%.6f')


if __name__ == '__main__':
    dataset_folder = 'D:/data/spades_lab/'
    prediction_set_folder = os.path.join(
        dataset_folder, 'DerivedCrossParticipants', 'location_matters', 'prediction_sets')
    summarize_predictions(prediction_set_folder)
