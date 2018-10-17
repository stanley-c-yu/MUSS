import pandas as pd
import os
from padar_converter.mhealth import dataset
import numpy as np
from functools import reduce, partial
from itertools import combinations
from padar_parallel.for_loop import ForLoop
from dask import delayed


def merge_placements(left_set, right_set):
    on = ['START_TIME', 'STOP_TIME', 'PID', 'SID', 'SENSOR_PLACEMENT']
    left_p = left_set['SENSOR_PLACEMENT'].values[0]
    right_p = right_set['SENSOR_PLACEMENT'].values[0]
    left_s = left_set['SID'].values[0]
    right_s = right_set['SID'].values[0]
    left_set.loc[:, 'SENSOR_PLACEMENT'] = left_p + '_' + right_p
    right_set.loc[:, 'SENSOR_PLACEMENT'] = left_p + '_' + right_p
    left_set.loc[:, 'SID'] = left_s + '_' + right_s
    right_set.loc[:, 'SID'] = left_s + '_' + right_s
    if '_' in left_p:
        left_suffix = ''
    else:
        left_suffix = '_' + left_p
    joined_feature_set = pd.merge(left_set, right_set, how='outer',
                                  on=on, suffixes=[left_suffix, '_' + right_p])
    return joined_feature_set


def merge_feature_and_class(left_set, right_set):
    on = ['START_TIME', 'STOP_TIME', 'PID']
    joined_set = pd.merge(left_set, right_set, how='outer',
                          on=on)
    joined_set = joined_set.dropna()
    return joined_set


@delayed
def merge_all_placements(feature_set, sensor_placements):
    placement_sets = []
    for placement in sensor_placements:
        placement_sets.append(
            feature_set.loc[feature_set['SENSOR_PLACEMENT'] == placement, :])
    return reduce(merge_placements, placement_sets)


@delayed
def filter_by_pids(feature_set, class_set, pids):
    feature_set = feature_set.loc[feature_set['PID'].isin(
        pids), :]
    class_set = class_set.loc[class_set['PID'].isin(pids), :]
    return (feature_set, class_set)


@delayed
def merge_joined_feature_and_class(bundle):
    feature_set = bundle[0]
    class_set = bundle[1]
    joined_set = feature_set.groupby(
        'PID').apply(merge_feature_and_class, class_set)
    return joined_set


@delayed
def filter_by_class_labels(joined_set, exclude_labels, class_name):
    return joined_set.loc[np.logical_not(
        joined_set[class_name].isin(exclude_labels)), :]


def save_datasets(joined_sets, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for joined_set in joined_sets:
        sensor_placements = joined_set['SENSOR_PLACEMENT']
        joined_data = joined_set['DATA']
        output_filepath = os.path.join(
            output_folder, '_'.join(sensor_placements) + '.dataset.csv')
        joined_data.to_csv(output_filepath, index=False)
        print('Saved ' + ','.join(sensor_placements))


def prepare_dataset(sensor_placements, feature_set, class_set, pids, output_folder=None, **kwargs):
    joined_feature_set = merge_all_placements(feature_set, sensor_placements)

    filtered_bundle = filter_by_pids(joined_feature_set, class_set, pids)

    joined_set = merge_joined_feature_and_class(filtered_bundle)

    joined_set = filter_by_class_labels(
        joined_set, ['Transition', 'Unknown'], 'ACTIVITY')

    joined_set = {'SENSOR_PLACEMENT': sensor_placements, 'DATA': joined_set}
    return joined_set


if __name__ == '__main__':
    input_folder = 'D:/data/spades_lab'
    output_folder = os.path.join(
        input_folder, 'DerivedCrossParticipants', 'location_matters')

    feature_set_file = os.path.join(
        output_folder, 'location_matters.feature.csv')
    class_set_file = os.path.join(output_folder, 'location_matters.class.csv')

    feature_set = delayed(pd.read_csv)(feature_set_file, parse_dates=[
        0, 1], infer_datetime_format=True)

    class_set = delayed(pd.read_csv)(class_set_file, parse_dates=[
        0, 1], infer_datetime_format=True)

    pids = dataset.get_pids(input_folder)
    placements = ['DW', 'NDW', 'DA', 'NDA', 'DH', 'NDH', 'DT']
    # placements = ['DW', 'NDW', 'DA']
    placement_combinations = list(reduce(
        lambda x, y: x + y, [list(combinations(placements, i)) for i in range(1, 8)]))

    experiment = ForLoop(placement_combinations, prepare_dataset,
                         feature_set=feature_set, class_set=class_set, pids=pids, output_folder=output_folder)

    experiment.compute(scheduler='processes')
    experiment.show_profiling()
    experiment.show_workflow('workflow.pdf')
    results = experiment.get_result()
    save_datasets(results, os.path.join(output_folder, 'validation_sets'))
