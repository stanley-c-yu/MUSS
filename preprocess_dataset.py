import numpy as np
import os
from glob import glob
import pandas as pd
from helper import log
from padar_converter.mhealth import dataset, fileio, dataframe
from padar_parallel.groupby import GroupBy, GroupByWindowing
from padar_parallel.grouper import MHealthGrouper
from padar_parallel.windowing import MhealthWindowing
from padar_parallel.join import join_as_dataframe
from padar_parallel.sort import sort_by_file_timestamp
from padar_features.transformations.accelerometer import orientation
from padar_features.libs.data_formatting.decorator import apply_on_accelerometer_dataframe
from clize import run
from dask import delayed
from functools import partial
import shutil


def save_to_file(data, metas, dataset_name):
    original_path = metas['original_path']
    new_path = original_path.replace(dataset_name, dataset_name + '_cleaned')
    new_dirs = os.path.dirname(new_path)
    os.makedirs(new_dirs, exist_ok=True)
    data.to_csv(
        new_path,
        header=True,
        index=False,
        float_format="%.3f",
    )
    print('saved ' + new_path)
    return data


def _preprocess_sensor_data(item, all_items, **kwargs):
    # get session boundaries
    metas = GroupBy.get_meta(item)

    # load data
    data_loader = delayed(fileio.load_sensor)
    loaded_data = data_loader(GroupBy.get_data(item))
    # apply offset mapping
    get_offset = partial(dataset.get_offset, offset_column=1)
    offset_in_secs = delayed(get_offset)(GroupBy.get_data(item))
    offset_data = delayed(dataframe.offset)(loaded_data, offset_in_secs)

    # apply orientation corrections
    orientation_correction = delayed(dataset.get_orientation_correction)(
        GroupBy.get_data(item))

    flip_and_swap = apply_on_accelerometer_dataframe(orientation.flip_and_swap)

    corrected_data = delayed(flip_and_swap)(
        offset_data,
        x_flip=orientation_correction[0],
        y_flip=orientation_correction[1],
        z_flip=orientation_correction[2])

    dataset_name = kwargs['dataset_name']

    corrected_data = delayed(save_to_file)(corrected_data, metas, dataset_name)

    return GroupBy.bundle(corrected_data, **metas)


def clean_sensor_data(input_folder,
                      output_folder,
                      debug_mode=True,
                      scheduler='processes'):

    sensor_files = glob(
        os.path.join(input_folder, '*', 'MasterSynced', '**',
                     'Actigraph*sensor.csv'),
        recursive=True)

    sensor_files = list(filter(dataset.is_pid_included, sensor_files))

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

    groupby.apply(
        _preprocess_sensor_data, dataset_name=os.path.basename(input_folder))

    groupby.final_join()

    groupby.compute(scheduler=scheduler).get_result()


def copy_annotation_files(input_folder, dataset_name):
    annotation_files = glob(
        os.path.join(input_folder, '*', 'MasterSynced', '**',
                     'SPADESInLab*annotation.csv'),
        recursive=True)
    print(annotation_files)
    annotation_files = list(filter(dataset.is_pid_included, annotation_files))
    for f in annotation_files:
        new_f = f.replace(dataset_name, dataset_name + '_cleaned')
        shutil.copyfile(f, new_f)
        print('copied to ' + new_f)


def copy_meta_files(input_folder, dataset_name):
    meta_files = [
        'location_mapping.csv', 'subjects.csv', 'muss_class_labels.csv',
        'pid_exceptions.csv', 'offset_mapping.csv',
        'orientation_corrections.csv'
    ]
    for f in meta_files:
        p = os.path.join(input_folder, 'DerivedCrossParticipants', f)
        new_p = p.replace(dataset_name, dataset_name + '_cleaned')
        os.makedirs(os.path.dirname(new_p), exist_ok=True)
        shutil.copyfile(p, new_p)
        print('copied to ' + new_p)


def main(input_folder, debug_mode=True, scheduler='processes'):
    if input_folder.endswith('/'):
        input_folder = input_folder[:-1]
    dataset_name = os.path.basename(input_folder)
    output_folder = input_folder.replace(dataset_name,
                                         dataset_name + '_cleaned')
    clean_sensor_data(input_folder, output_folder, debug_mode, scheduler)
    copy_annotation_files(input_folder, dataset_name)
    copy_meta_files(input_folder, dataset_name)


if __name__ == '__main__':
    # input_folder = os.path.join(
    #     os.path.expanduser('~'), 'Projects/data/spades_lab')
    # input_folder = 'D:/data/mini_mhealth_dataset/'
    # # input_folder = 'D:/data/spades_lab'
    # scheduler = 'sync'
    # print(input_folder)
    # main(input_folder, scheduler=scheduler)
    run(main)
