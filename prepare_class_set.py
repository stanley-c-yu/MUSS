import numpy as np
import os
from glob import glob
import pandas as pd
from helper import log, utils
from padar_converter.mhealth import dataset, fileio, dataframe
from padar_converter.annotation.data_format import to_mutually_exclusive
from padar_parallel.groupby import GroupBy, GroupByWindowing
from padar_parallel.grouper import MHealthGrouper
from padar_parallel.windowing import MhealthWindowing
from padar_parallel.join import join_as_dataframe
from padar_parallel.sort import sort_by_file_timestamp
from clize import run
from dask import delayed
from helper.annotation_processor import ClassLabeler
from helper.utils import generate_run_folder


def preprocess_annotations(item, all_items, **kwargs):
    # get session boundaries
    metas = GroupBy.get_meta(item)

    # load data
    data_loader = delayed(fileio.load_annotation)
    loaded_data = data_loader(GroupBy.get_data(item))

    return GroupBy.bundle(loaded_data, **metas)


def get_class_map(input_folder, annotation_files, scheduler='synchronous'):
    groupby = GroupBy(annotation_files,
                      **MhealthWindowing.make_metas(annotation_files))

    grouper = MHealthGrouper(annotation_files)
    groups = [grouper.pid_group(), grouper.annotator_group()]

    groupby.split(
        *groups, ingroup_sortkey_func=sort_by_file_timestamp, descending=False)

    groupby.apply(preprocess_annotations)
    groupby.final_join(delayed(join_as_dataframe))

    merged_annotations = groupby.compute(scheduler=scheduler).get_result()
    splitted_annotations = to_mutually_exclusive(merged_annotations)
    class_label_set = os.path.join(input_folder, 'DerivedCrossParticipants',
                                   'muss_class_labels.csv')
    class_map = ClassLabeler.from_annotation_set(
        splitted_annotations, class_label_set, interval=12.8)
    return class_map


def get_class_set(annotation_files, class_map, scheduler='synchronous'):
    groupby = GroupBy(annotation_files,
                      **MhealthWindowing.make_metas(annotation_files))

    grouper = MHealthGrouper(annotation_files)
    groups = [grouper.pid_group(), grouper.annotator_group()]

    groupby.split(
        *groups,
        group_types=['PID', 'ANNOTATOR'],
        ingroup_sortkey_func=sort_by_file_timestamp,
        descending=False)
    groupby.apply(preprocess_annotations)
    groupby.apply(
        convert_annotations, interval=12.8, step=12.8, class_map=class_map)
    groupby.final_join(delayed(join_as_dataframe))

    class_set = groupby.compute(scheduler=scheduler).get_result()
    return (class_set, groupby)


@delayed
@MhealthWindowing.groupby_windowing('annotation')
def convert_annotations(df, **kwargs):
    class_map = kwargs['class_map']
    interval = kwargs['interval']
    if df.empty:
        return class_map.loc[class_map['ANNOTATION_LABELS'] == 'empty', :]
    df.iloc[:, 3] = df.iloc[:, 3].str.lower()
    df = df.loc[(df.iloc[:, 3] != 'wear on') | (df.iloc[:, 3] != 'wearon'), :]

    labels = df.iloc[:, 3].unique()
    labels.sort()
    labels = ' '.join(labels).lower().replace('wear on', '').replace(
        'wearon', '').strip()

    # filter if it does not cover the entire 12.8s
    df_durations = df.groupby(df.columns[3]).apply(
        lambda rows: np.sum(rows.iloc[:, 2] - rows.iloc[:, 1]))
    if not np.all(df_durations.values >= np.timedelta64(
            int(interval * 0.8 * 1000), 'ms')):
        matched_classes = class_map.loc[class_map['ANNOTATION_LABELS'] ==
                                        'empty', :]
        matched_classes = pd.DataFrame(
            data=[[labels] + ['Transition'] * 5],
            columns=matched_classes.columns,
            index=[0])
        return matched_classes
    # get labels from class map
    matched_classes = class_map.loc[class_map['ANNOTATION_LABELS'] ==
                                    labels, :]

    if matched_classes.empty:
        matched_classes = pd.DataFrame(
            data=[[labels] + ['Unknown'] * 5],
            columns=matched_classes.columns,
            index=[0])
    return matched_classes


def prepare_class_set(input_folder, *, debug=False, scheduler='processes'):
    """Compute class set for "Location Matters" paper by Tang et al.

    Process the given annotations (stored in mhealth format) and generate class set file in csv format along with a profiling report and class set conversion  pipeline diagram.

    :param input_folder: Folder path of input raw dataset
    :param debug: Use this flag to output results to 'debug_run' folder
    :param scheduler: 'processes': Use multi-core processing;
                      'threads': Use python threads (not-in-parallel);
                      'sync': Use a single thread in sequential order
    """

    input_folder = utils.strip_path(input_folder)
    output_folder = utils.generate_run_folder(input_folder, debug=debug)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    annotation_files = glob(
        os.path.join(input_folder, '*', 'MasterSynced', '**',
                     'SPADESInLab*annotation.csv'),
        recursive=True)

    # get class map
    class_map = get_class_map(
        input_folder, annotation_files, scheduler=scheduler)
    classmap_filepath = os.path.join(output_folder, 'muss.classmap.csv')
    class_map.to_csv(classmap_filepath, index=False)

    class_set, groupby = get_class_set(
        annotation_files, class_map=class_map, scheduler=scheduler)

    classset_filepath = os.path.join(output_folder, 'muss.class.csv')

    profiling_filepath = os.path.join(output_folder,
                                      'classset_computation_profiling.html')
    workflow_filepath = os.path.join(output_folder,
                                     'classset_computation_workflow.pdf')
    class_set.to_csv(classset_filepath, index=True)
    groupby.show_profiling(file_path=profiling_filepath)
    try:
        groupby.visualize_workflow(filename=workflow_filepath)
    except Exception as e:
        print(e)
        print('skip generating workflow pdf')


if __name__ == '__main__':
    # input_folder = os.path.join(
    #     os.path.expanduser('~'), 'Projects/data/mini-mhealth-dataset')
    # input_folder = 'D:/data/mini_mhealth_dataset_cleaned/'
    # input_folder = 'D:/data/spades_lab/'
    # output_folder = generate_run_folder(input_folder, debug=False)

    # scheduler = 'processes'
    # prepare_class_set(input_folder, scheduler=scheduler)
    run(prepare_class_set)
