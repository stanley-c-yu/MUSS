import numpy as np
import os
from glob import glob
import pandas as pd
from helper import log, preprocess
from padar_converter.mhealth import dataset, fileio, dataframe
from padar_parallel.groupby import GroupBy, GroupByWindowing
from padar_parallel.grouper import MHealthGrouper
from padar_parallel.windowing import MhealthWindowing
from padar_parallel.join import join_as_dataframe
from padar_features.feature_extractor import FeatureExtractor
from padar_features.feature_set import FeatureSet
from padar_features.transformations.accelerometer import orientation
from clize import run
from dask import delayed


def sort_func(item):
    return dataset.get_file_timestamp(GroupBy.get_data(item))


def preprocess_data(item, all_items, **kwargs):

    # get session boundaries
    metas = GroupBy.get_meta(item)

    # load data
    data_loader = delayed(fileio.load_sensor)
    loaded_data = data_loader(GroupBy.get_data(item))

    # apply offset mapping
    offset_in_secs = delayed(preprocess.get_offset)(GroupBy.get_data(item))
    offset_data = delayed(dataframe.offset)(loaded_data, offset_in_secs)

    # apply orientation corrections
    orientation_corrector = delayed(preprocess.get_orientation_correction)
    orientation_correction = orientation_corrector(GroupBy.get_data(item))

    corrected_data = delayed(orientation.flip_and_swap)(
        offset_data.values, x_flip=orientation_correction[0], y_flip=orientation_correction[1], z_flip=orientation_correction[2])

    return GroupBy.bundle(corrected_data, **metas)


@delayed
@MhealthWindowing.groupby_windowing('sensor')
def compute_features(df, **kwargs):
    return FeatureSet.location_matters(df.values, **kwargs)


def prepare_feature_set(input_folder, output_folder, debug_mode=True, sampling_rate=80, scheduler='processes'):
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
    logger = log.get_logger(output_folder,
                            'location_matters.prepare_feature_set',
                            debug=debug_mode)

    sensor_files = glob(os.path.join(
        input_folder, '*', 'MasterSynced', '**', 'Actigraph*sensor.csv'), recursive=True)

    sensor_files = list(filter(exclude.exclude_pid, sensor_files))

    groupby = GroupBy(
        sensor_files, **MhealthWindowing.make_metas(sensor_files))

    grouper = MHealthGrouper(sensor_files)
    groups = [
        grouper.pid_group(),
        grouper.sid_group(),
        grouper.auto_init_placement_group()
    ]

    groupby.split(
        grouper.pid_group(),
        grouper.sid_group(),
        ingroup_sortkey_func=sort_func,
        descending=False)

    groupby.apply(preprocess_data)

    groupby.apply(compute_features, interval=12.8,
                  step=12.8, sr=sampling_rate)

    groupby.final_join(delayed(join_as_dataframe))

    result = groupby.compute(
        scheduler=scheduler).get_result()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    feature_filepath = os.path.join(
        output_folder, 'location_matters.feature.csv')
    profiling_filepath = os.path.join(
        output_folder, 'feature_computation_profiling.html')
    workflow_filepath = os.path.join(
        output_folder, 'feature_computation_workflow.pdf')
    result.to_csv(output_path, float_format='%.9f', index=True)
    groupby.show_profiling(file_path=profiling_filepath)
    groupby.visualize_workflow(filename=workflow_filepath)


if __name__ == '__main__':
    run(prepare_feature_set)
