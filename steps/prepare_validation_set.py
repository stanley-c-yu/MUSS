import pandas as pd
import os
from padar_converter.mhealth import dataset
import numpy as np
from functools import reduce, partial
from itertools import combinations, product
from padar_parallel.for_loop import ForLoop
from dask import delayed
from helper import utils
from clize import run
from sklearn.utils import shuffle
import logging


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
    joined_feature_set = pd.merge(left_set, right_set, how='outer', on=on)
    return joined_feature_set


def merge_feature_and_class(left_set, right_set):
    on = ['START_TIME', 'STOP_TIME', 'PID']
    joined_set = pd.merge(left_set, right_set, how='outer', on=on)
    joined_set = joined_set.dropna()
    return joined_set


@delayed
def merge_all_placements(feature_set, sensor_placements):
    non_feature_columns = [
        'START_TIME', 'STOP_TIME', 'PID', 'SID', 'SENSOR_PLACEMENT'
    ]
    placement_sets = []
    for placement in sensor_placements:
        placement_set = feature_set.loc[feature_set['SENSOR_PLACEMENT'] ==
                                        placement, :]
        if placement_set.empty:
            continue
        placement_set.set_index(non_feature_columns, inplace=True)
        feature_columns = placement_set.columns
        feature_columns = [col + '_' + placement for col in feature_columns]
        placement_set.columns = feature_columns
        placement_set.reset_index(drop=False, inplace=True)
        placement_sets.append(placement_set)
    if len(placement_sets) == 0:
        return None
    else:
        return reduce(merge_placements, placement_sets)


@delayed
def filter_by_pids(feature_set, pids, class_set=None):
    if feature_set is None:
        return (feature_set, class_set)
    feature_set = feature_set.loc[feature_set['PID'].isin(pids), :]
    if class_set is not None:
        class_set = class_set.loc[class_set['PID'].isin(pids), :]
    return (feature_set, class_set)


@delayed
def filter_by_feature_type(feature_set, feature_type):
    if feature_set is None:
        return None
    feature_cols = feature_set.columns.values[5:].tolist()
    if feature_type == 'M':
        feature_cols = list(
            filter(lambda col: 'ANGLE' not in col, feature_cols))
    elif feature_type == 'O':
        feature_cols = list(filter(lambda col: 'ANGLE' in col, feature_cols))
    selected_cols = feature_set.columns.values[:5].tolist() + feature_cols
    result = feature_set[selected_cols]
    result.insert(5, 'FEATURE_TYPE', feature_type)
    return result


@delayed
def merge_joined_feature_and_class(bundle):
    feature_set = bundle[0]
    class_set = bundle[1]
    if feature_set is not None and class_set is None:
        return feature_set
    elif feature_set is None:
        return None
    else:
        joined_set = feature_set.groupby('PID').apply(merge_feature_and_class,
                                                      class_set)
        return joined_set


@delayed
def filter_by_class_labels(joined_set, exclude_labels, class_name):
    if joined_set is None:
        return None
    elif class_name in joined_set:
        return joined_set.loc[np.logical_not(joined_set[class_name].
                                             isin(exclude_labels)), :]
    else:
        return joined_set


@delayed
def save_validation_set(joined_set, sensor_placements, feature_type,
                        output_folder):
    if joined_set is None:
        return {}
    os.makedirs(output_folder, exist_ok=True)
    output_filepath = os.path.join(
        output_folder,
        '_'.join(sensor_placements) + '.' + feature_type + '.dataset.csv')
    joined_set.to_csv(output_filepath, index=False)
    print('Saved ' + ','.join(sensor_placements) + ' ' + feature_type)
    joined_set = {
        'SENSOR_PLACEMENT': sensor_placements,
        'DATA': joined_set,
        'FEATURE_TYPE': feature_type
    }
    return joined_set


@delayed
def distribute_nonwear_to_pids(nonwear_set, pids):
    shuffled_nonwear_set = shuffle(nonwear_set)
    n_samples = nonwear_set.shape[0]
    n_pids = len(pids)
    n_samples_per_pid = int(np.floor(float(n_samples) / n_pids))
    for i in range(0, n_pids):
        start_index = 0 + i * n_samples_per_pid
        stop_index = n_samples_per_pid * (i + 1)

        shuffled_nonwear_set.iloc[start_index:stop_index, 2] = pids[i]
    for i in range(stop_index, n_samples):
        shuffled_nonwear_set.iloc[i:(i + 1), 2] = pids[i - stop_index]
    return shuffled_nonwear_set


def set_nonwear_classes(joined_set):
    return joined_set.fillna(value='Nonwear')


def prepare_dataset(input_bundles,
                    pids,
                    feature_set_file,
                    class_set_file=None,
                    output_folder=None,
                    nonwear_set_file=None,
                    target=None,
                    **kwargs):

    sensor_placements = input_bundles[0]

    feature_type = input_bundles[1]

    feature_set = delayed(pd.read_csv)(
        feature_set_file, parse_dates=[0, 1], infer_datetime_format=True)

    if class_set_file is not None:
        class_set = delayed(pd.read_csv)(
            class_set_file, parse_dates=[0, 1], infer_datetime_format=True)
    else:
        class_set = None

    if nonwear_set_file is not None:
        nonwear_set = delayed(pd.read_csv)(
            nonwear_set_file, parse_dates=[0, 1]
        )
    else:
        nonwear_set = None

    joined_feature_set = merge_all_placements(feature_set, sensor_placements)

    filtered_feature_set = filter_by_feature_type(joined_feature_set,
                                                  feature_type)

    filtered_bundle = filter_by_pids(filtered_feature_set, pids, class_set)

    joined_set = merge_joined_feature_and_class(filtered_bundle)
    if target is None:
        joined_set = filter_by_class_labels(
            joined_set, ['Transition', 'Unknown'], 'MUSS_22_ACTIVITIES')
    else:
        joined_set = filter_by_class_labels(
            joined_set, ['Transition', 'Unknown'], target)

    if nonwear_set is not None:
        distributed_nonwear_set = distribute_nonwear_to_pids(nonwear_set, pids)
        joined_set = delayed(pd.concat)([joined_set, distributed_nonwear_set])
        joined_set = set_nonwear_classes(joined_set)

    return save_validation_set(joined_set, sensor_placements, feature_type, output_folder)


def main(input_folder, *, output_folder=None, sites=None, feature_types=None, include_nonwear=False, target=None, debug=False, scheduler='processes', profiling=True, force=True):
    """Prepare validation sets

    :param input_folder: Folder path of input raw dataset
    :param output_folder: auto path if None
    :param debug: Use this flag to output results to 'debug_run' folder
    :param scheduler: 'processes': Use multi-core processing;
                      'threads': Use python threads (not-in-parallel);
                      'sync': Use a single thread in sequential order
    :param profiling: use profiling or not.
    """
    if output_folder is None:
        output_folder = utils.generate_run_folder(input_folder, debug=debug)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if include_nonwear:
        if target is None:
            datasets_folder = os.path.join(
                output_folder, 'datasets_with_nonwear')
        else:
            datasets_folder = os.path.join(
                output_folder, target + '_datasets_with_nonwear')
    else:
        if target is None:
            datasets_folder = os.path.join(output_folder, 'datasets')
        else:
            datasets_folder = os.path.join(output_folder, target + '_datasets')

    if not force and os.path.exists(datasets_folder):
        logging.info('Datasets folder exits, skip regenerating...')
        return datasets_folder

    feature_set_file = os.path.join(output_folder, 'muss.feature.csv')
    class_set_file = os.path.join(output_folder, 'muss.class.csv')
    if not os.path.exists(class_set_file):
        class_set_file = None
    profiling_filepath = os.path.join(output_folder,
                                      'dataset_computation_profiling.html')
    workflow_filepath = os.path.join(output_folder,
                                     'dataset_computation_workflow.pdf')

    pids = dataset.get_pids(input_folder)
    if sites is None or feature_types is None:
        placements = ['DW', 'NDW', 'DA', 'NDA', 'DH', 'NDH', 'DT']
        feature_types = ['M', 'O', 'MO']
    else:
        placements = sites.split(',')
        feature_types = feature_types.split(',')
    placement_combinations = list(
        reduce(lambda x, y: x + y,
               [list(combinations(placements, i)) for i in range(1, 8)]))
    input_bundles = product(placement_combinations, feature_types)

    if include_nonwear:
        nonwear_set_file = "nonwear.feature.csv"
    else:
        nonwear_set_file = None

    experiment = ForLoop(
        input_bundles,
        prepare_dataset,
        feature_set_file=feature_set_file,
        class_set_file=class_set_file,
        nonwear_set_file=nonwear_set_file,
        target=target,
        pids=pids,
        output_folder=datasets_folder)

    experiment.compute(scheduler=scheduler, profiling=profiling)
    if profiling:
        try:
            experiment.show_workflow(workflow_filepath)
        except Exception as e:
            print(e)
            print('skip generating workflow pdf')
        experiment.show_profiling(profiling_filepath)
    return datasets_folder


if __name__ == '__main__':
    run(main)
    # main('D:/data/muss_data/', debug=True, scheduler='sync',
    #      sites='DW', feature_types='MO', include_nonwear=True)
