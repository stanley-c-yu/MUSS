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
from padar_converter.dataset import spades


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


def get_class_row(matched_class, labels, class_map):
    n = class_map.shape[1]
    if matched_class == 'Unknown' or matched_class == 'Transition':
        matched_classes = pd.DataFrame(
            data=[[labels] + [matched_class] * n],
            columns=['ANNOTATION_LABELS'] + class_map.columns.values.tolist(),
            index=[0])
    else:
        matched_classes = pd.DataFrame(
            data=[[labels] + class_map.loc[class_map['ACTIVITY']
                                           == matched_class, :].values[0].tolist()],
            columns=['ANNOTATION_LABELS'] + class_map.columns.values.tolist(),
            index=[0]
        )
    return matched_classes


@delayed
@MhealthWindowing.groupby_windowing('annotation')
def convert_annotations(df, **kwargs):
    class_map = kwargs['class_map']
    interval = kwargs['interval']
    original_path = kwargs['original_path']
    pid = dataset.get_pid(original_path)
    if df.empty:
        matched_classes = get_class_row('Unknown', '', class_map)
        return matched_classes
    df.iloc[:, 3] = df.iloc[:, 3].str.lower()
    df = df.loc[(df.iloc[:, 3] != 'wear on') & (df.iloc[:, 3] != 'wearon'), :]
    if df.empty:
        matched_classes = get_class_row('Unknown', '', class_map)
        return matched_classes
    labels = df.iloc[:, 3].unique()
    labels.sort()
    labels = ' '.join(labels).lower().strip()

    # filter if it does not cover the entire 12.8s
    df_durations = df.groupby(df.columns[3]).apply(
        lambda rows: np.sum(rows.iloc[:, 2] - rows.iloc[:, 1]))
    if not np.all(df_durations.values >= np.timedelta64(
            int(interval * 1000), 'ms')):
        matched_classes = get_class_row('Transition', labels, class_map)
        return matched_classes
    # get labels from class map
    st = np.min(df.iloc[:, 1])
    et = np.max(df.iloc[:, 2])
    matched_class = spades.to_inlab_activity_labels(labels, pid, st, et)
    matched_classes = get_class_row(matched_class, labels, class_map)
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

    class_map_file = os.path.join(input_folder, 'DerivedCrossParticipants',
                                  'muss_class_labels.csv')
    class_map = pd.read_csv(class_map_file)
    class_set, groupby = get_class_set(
        annotation_files, class_map=class_map, scheduler=scheduler)

    class_set_unique = class_set.drop_duplicates(
        subset=['ANNOTATION_LABELS', 'ACTIVITY'], keep='first')

    classset_filepath = os.path.join(output_folder, 'muss.class.csv')
    classset_unique_filepath = os.path.join(output_folder, 'muss.classmap.csv')
    profiling_filepath = os.path.join(output_folder,
                                      'classset_computation_profiling.html')
    workflow_filepath = os.path.join(output_folder,
                                     'classset_computation_workflow.pdf')
    class_set.to_csv(classset_filepath, index=True)
    class_set_unique.to_csv(classset_unique_filepath, index=True)
    groupby.show_profiling(file_path=profiling_filepath)
    try:
        groupby.visualize_workflow(filename=workflow_filepath)
    except Exception as e:
        print(e)
        print('skip generating workflow pdf')


if __name__ == '__main__':
    # input_folder = 'D:/data/muss_data/'
    # scheduler = 'processes'
    # prepare_class_set(input_folder, scheduler=scheduler, debug=True)
    run(prepare_class_set)
