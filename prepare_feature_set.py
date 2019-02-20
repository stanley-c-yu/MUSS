import numpy as np
import os
from glob import glob
import pandas as pd
from helper import log, utils
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


def prepare_feature_set(input_folder,
                        output_folder,
                        debug_mode=True,
                        sampling_rate=80,
                        scheduler='processes'):
    """Compute feature set for "Location Matters" paper by Tang et al.

    Process the given raw dataset (stored in mhealth format) and generate feature set file in csv format along with a profiling report and feature computation pipeline diagram.

    :param input_folder: Folder path of input raw dataset
    :param output_folder: Folder path of output feature set files and other computation reports
    :param debug_mode: If true, output debug messages in log file
    :param sampling_rate: The sampling rate of the raw accelerometer data in Hz
    :param scheduler: 'processes': Use multi-core processing; 
                      'threads': Use python threads (not-in-parallel)
                      'synchronous': Use a single thread in sequential order
    """

    sensor_files = glob(
        os.path.join(input_folder, '*', 'MasterSynced', '**',
                     'Actigraph*sensor.csv'),
        recursive=True)

    groupby = GroupBy(sensor_files,
                      **MhealthWindowing.make_metas(sensor_files))

    grouper = MHealthGrouper(sensor_files)
    groups = [
        grouper.pid_group(),
        grouper.sid_group(),
        grouper.auto_init_placement_group()
    ]

    groupby.split(
        *groups,
        group_types=['PID', 'SID', 'SENSOR_PLACEMENT'],
        ingroup_sortkey_func=sort_by_file_timestamp,
        descending=False)

    groupby.apply(load_data)

    groupby.apply(compute_features, interval=12.8, step=12.8, sr=sampling_rate)

    groupby.final_join(delayed(join_as_dataframe))

    result = groupby.compute(scheduler=scheduler).get_result()

    # rename placements
    result = result.reset_index()
    result.loc[:,
               'SENSOR_PLACEMENT'] = result.loc[:, 'SENSOR_PLACEMENT'].apply(
                   dataset.get_placement_abbr)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    feature_filepath = os.path.join(output_folder, 'muss.feature.csv')
    profiling_filepath = os.path.join(output_folder,
                                      'feature_computation_profiling.html')
    workflow_filepath = os.path.join(output_folder,
                                     'feature_computation_workflow.pdf')
    result.to_csv(feature_filepath, float_format='%.9f', index=False)
    groupby.show_profiling(file_path=profiling_filepath)
    groupby.visualize_workflow(filename=workflow_filepath)


if __name__ == '__main__':
    input_folder = os.path.join(
        os.path.expanduser('~'), 'Projects/data/spades_lab')
    input_folder = 'D:/data/mini_mhealth_dataset_cleaned'
    # input_folder = 'D:/data/spades_lab'
    output_folder = utils.generate_run_folder(input_folder, debug=False)
    sampling_rate = 80
    scheduler = 'processes'
    print(input_folder)
    prepare_feature_set(
        input_folder,
        output_folder,
        sampling_rate=sampling_rate,
        scheduler=scheduler)
    # run(prepare_feature_set)
