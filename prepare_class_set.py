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
from clize import run
from dask import delayed
from helper.annotation_processor import annotation_splitter, class_mapping


def sort_func(item):
    return dataset.get_file_timestamp(GroupBy.get_data(item))


def preprocess_annotations(item, all_items, **kwargs):
    # get session boundaries
    metas = GroupBy.get_meta(item)

    # load data
    data_loader = delayed(fileio.load_annotation)
    loaded_data = data_loader(GroupBy.get_data(item))

    return GroupBy.bundle(loaded_data, **metas)


def get_class_map(annotation_files, scheduler='synchronized'):
    groupby = GroupBy(
        annotation_files, **MhealthWindowing.make_metas(annotation_files))

    grouper = MHealthGrouper(annotation_files)
    groups = [
        grouper.pid_group(),
        grouper.annotator_group()
    ]

    groupby.split(
        *groups,
        ingroup_sortkey_func=sort_func,
        descending=False)

    groupby.apply(preprocess_annotations)
    groupby.final_join(delayed(join_as_dataframe))

    merged_annotations = groupby.compute(
        scheduler=scheduler).get_result()
    splitted_annotations = annotation_splitter(merged_annotations)
    class_map = class_mapping(splitted_annotations)
    return class_map

def get_class_set(annotation_files, class_map, scheduler='synchronized'):
    groupby = GroupBy(
        annotation_files, **MhealthWindowing.make_metas(annotation_files))

    grouper = MHealthGrouper(annotation_files)
    groups = [
        grouper.pid_group(),
        grouper.annotator_group()
    ]

    groupby.split(
        *groups,
        ingroup_sortkey_func=sort_func,
        descending=False)

    groupby.apply(convert_annotations, interval=12.8, step=12.8, class_map=class_map)
    groupby.final_join(delayed(join_as_dataframe))

    class_set = groupby.compute(
        scheduler=scheduler).get_result()
    return class_set

@delayed
@MhealthWindowing.groupby_windowing('annotation')
def convert_annotations(df, **kwargs):
    sadf


def prepare_class_set(input_folder, output_folder, debug_mode=True, scheduler='processes'):
    """Compute class set for "Location Matters" paper by Tang et al.

    Process the given annotations (stored in mhealth format) and generate class set file in csv format along with a profiling report and class set conversion  pipeline diagram.

    :param input_folder: Folder path of input raw dataset
    :param output_folder: Folder path of output feature set files and other computation reports
    :param debug_mode: If true, output debug messages in log file
    :param scheduler: 'processes': Use multi-core processing; 
                      'threads': Use python threads (not-in-parallel)
                      'synchronous': Use a single thread in sequential order
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    logger = log.get_logger(output_folder,
                            'location_matters.prepare_class_set',
                            debug=debug_mode)

    annotation_files = glob(os.path.join(input_folder, '*', 'MasterSynced', '**', 'SPADESInLab*annotation.csv'), recursive=True)

    annotation_files = list(filter(preprocess.include_pid, annotation_files))

    # get class map
    class_map = get_class_map(annotation_files, scheduler=scheduler)
    classmap_filepath = os.path.join(
        output_folder, 'location_matters.classmap.csv')
    class_map.to_csv(classmap_filepath, index=True)

    class_set = get_class_set(annotation_files, class_map=class_map, scheduler=scheduler)

    
    classset_filepath = os.path.join(
        output_folder, 'location_matters.class.csv')

    
    profiling_filepath = os.path.join(
        output_folder, 'classset_computation_profiling.html')
    workflow_filepath = os.path.join(
        output_folder, 'classset_computation_workflow.pdf')
    class_set.to_csv(classset_filepath, index=True)

if __name__ == '__main__':
    input_folder = os.path.join(os.path.expanduser('~'), 'Projects/data/mini-mhealth-dataset')
    output_folder = os.path.join(
        input_folder, 'DerivedCrossParticipants', 'location_matters')
    scheduler = 'processes'
    print(input_folder)
    prepare_class_set(input_folder, output_folder, scheduler=scheduler)
    # run(prepare_feature_set)
