import numpy as np
import os
from glob import glob
import pandas as pd
from padar_converter.mhealth import dataset, fileio, dataframe
from padar_parallel.groupby import GroupBy, GroupByWindowing
from padar_parallel.grouper import MHealthGrouper
from padar_parallel.windowing import MhealthWindowing
from padar_parallel.join import join_as_dataframe
from padar_parallel.sort import sort_by_file_timestamp
from padar_features.feature_extractor import FeatureExtractor
from padar_features.feature_set import FeatureSet
from padar_features.transformations.accelerometer import orientation
from padar_features.libs.data_formatting.decorator import apply_on_accelerometer_dataframe
from clize import run
from dask import delayed
from functools import partial
import pickle


MODEL_FILE = "DW.MO.classic_seven_activities_model.pkl"


def strip_path(path):
    if path.endswith('/'):
        return path[:-1]


def find_sensor_files(input_folder, pids='*'):
    return glob(
        os.path.join(input_folder, pids, 'MasterSynced', '**',
                     'Actigraph*sensor.csv'),
        recursive=True)


def load_data(item, all_items, **kwargs):
    # get session boundaries
    metas = GroupBy.get_meta(item)

    # load data
    data_loader = delayed(fileio.load_sensor)
    loaded_data = data_loader(GroupBy.get_data(item))

    return GroupBy.bundle(loaded_data, **metas)


@delayed
@MhealthWindowing.groupby_windowing('sensor')
def compute_features(df, **kwargs):
    return FeatureSet.location_matters(df.values[:, 1:], **kwargs)


def get_feature_set(sensor_files, sampling_rate=80):
    groupby = GroupBy(sensor_files,
                      **MhealthWindowing.make_metas(sensor_files))
    grouper = MHealthGrouper(sensor_files)
    groups = [
        grouper.pid_group()
    ]

    groupby.split(
        *groups,
        group_types=['PID'],
        ingroup_sortkey_func=sort_by_file_timestamp,
        descending=False)

    groupby.apply(load_data)
    groupby.apply(compute_features, interval=12.8, step=12.8, sr=sampling_rate)
    groupby.final_join(delayed(join_as_dataframe))
    feature_set = groupby.compute(scheduler='processes').get_result()
    feature_columns = feature_set.columns
    feature_columns = [col + '_' + 'DW' for col in feature_columns]
    feature_set.columns = feature_columns
    feature_set = feature_set.reset_index()
    return feature_set


def make_input_matrix(feature_df, model_bundle):
    feature_order = model_bundle['feature_order']
    ordered_df = feature_df.loc[:, feature_order]
    X = ordered_df.values
    scaled_X = model_bundle['scaler'].transform(X)
    return scaled_X


def get_prediction_set(feature_set):
    feature_set = feature_set.dropna()
    indexed_feature_df = feature_set.set_index([
        'START_TIME', 'STOP_TIME', 'PID'
    ])
    p_df = feature_set
    with open(MODEL_FILE, 'rb') as mf:
        model_bundle = pickle.load(mf)
        X = make_input_matrix(indexed_feature_df, model_bundle)
        try:
            predicted_labels = model_bundle['model'].predict(X)
        except:
            predicted_labels = X.shape[0] * [np.nan]
        p_df['PREDICTION'] = predicted_labels
    return p_df


def main(input_folder, pids='*', sampling_rate=80):
    input_folder = strip_path(input_folder)
    sensor_files = find_sensor_files(input_folder, pids=pids)
    feature_set = get_feature_set(sensor_files, sampling_rate=sampling_rate)
    prediction_set = get_prediction_set(feature_set)
    print(prediction_set)


if __name__ == '__main__':
    main('D:/data/MDCAS/', pids='SPADES_17')
