import glob
import pandas as pd
from helper.svm_model import svm_model, rf_model
from helper.utils import generate_run_folder
import os
import pickle
from clize import run
import numpy as np


def main(input_folder,
         *,
         output_folder=None,
         debug=False,
         targets='MDCAS,RIAR_17_ACTIVITIES',
         feature_set='MO',
         sites='DW,DA', include_nonwear=False, model_type='svm'):
    """Train and save a model using one of the validation datasets

    :param input_folder: Folder path of input raw dataset.
    :param output_folder: Auto path if None.
    :param debug: Use this flag to output results to 'debug_run' folder.
    :param targets: The list of groups of class labels, separated by ','.

    :param feature_set: Choose the type of feature set to be used.

                Allowed feature sets:
                'MO': use 'motion + orientation' features;
                'M': use 'motion" features;
                'O': use 'orientation' features.
    :param sites: String of list of placements of sensors, separated by ','.

                Allowed placement codes,
                'DW': dominant wrist;
                'DA': dominant ankle;
                'DT': dominant thigh;
                'DH': dominant hip;
                'NDW': nondominant wrist;
                'NDA': nondominant ankle;
                'NDH': nondominant hip.
    """
    if output_folder is None:
        output_folder = generate_run_folder(input_folder, debug=debug)
    sites = sites.split(',')
    targets = targets.split(',')
    predefined_targets = [
        'MUSS_3_POSTURES', 'MUSS_22_ACTIVITIES', 'RIAR_17_ACTIVITIES',
        'MDCAS'
    ]
    predefined_sites = ['DW', 'DA', 'DT', 'DH', 'NDW', 'NDA', 'NDH']
    if feature_set not in ['MO', 'O', 'M']:
        raise Exception(
            "Input parameter 'feature_set' should be one of 'MO', 'O' or 'M'.")
    for site in sites:
        if site not in predefined_sites:
            raise Exception("Input parameter 'sites' should be one of " +
                            ','.join(predefined_sites))
    for target in targets:
        if target not in predefined_targets:
            raise Exception("Input parameter 'targets' should be one of " +
                            ','.join(predefined_targets))
    class_mapping = grouping_file(input_folder)
    suffix = '_with_nonwear' if include_nonwear else ''
    for target in targets:
        dataset_folder = os.path.join(output_folder, target + '_datasets' + suffix)
        train_and_save_model(dataset_folder, sites=sites,
                         feature_set=feature_set, target=target, class_mapping=class_mapping, model_type=model_type)
    return os.path.join(os.path.dirname(dataset_folder), 'models')


def grouping_file(input_folder):
    class_label_set = os.path.join(input_folder, 'MetaCrossParticipants',
                                   'muss_class_labels.csv')
    class_mapping = pd.read_csv(class_label_set)
    return class_mapping


def train_and_save_model(dataset_folder,
                         target='MUSS_3_POSTURES',
                         sites=['DW', 'DA'],
                         feature_set='MO',
                         class_mapping=None, model_type='svm'):
    validation_set_files = glob.glob(
        os.path.join(dataset_folder, '*.dataset.csv'), recursive=True)
    selected_file = None
    model_folder = os.path.join(os.path.dirname(dataset_folder), 'models')
    os.makedirs(model_folder, exist_ok=True)
    for f in validation_set_files:
        fname = os.path.basename(f)
        file_sites = fname.split('.')[0].split('_')
        file_sites.sort()
        sites.sort()
        file_fs = fname.split('.')[1]
        if file_sites == sites and feature_set == file_fs:
            selected_file = f
            break
    if selected_file:
        dataset = pd.read_csv(
            selected_file, parse_dates=[0, 1], infer_datetime_format=True)
        model_path = os.path.join(
            model_folder,
            os.path.basename(selected_file).replace(
                'dataset',
                target.lower() + '_model').replace('csv', 'pkl'))

        model, scaler, training_accuracy, feature_order = train_model(
            dataset, get_train_target(target), model_type=model_type)
        save_model(model_path, target, model, scaler, training_accuracy,
                    feature_order, class_mapping)


def get_train_target(target):
    if target == 'MUSS_22_ACTIVITIES' or target == 'MUSS_3_POSTURES' or target == 'MDCAS' or target == 'RIAR_17_ACTIVITIES':
        return target
    else:
        return 'ACTIVITY'


def train_model(dataset, train_target, model_type='svm'):
    index_cols = [
        "START_TIME", "STOP_TIME", "PID", "SID", "SENSOR_PLACEMENT",
        "FEATURE_TYPE", "ANNOTATOR", "ANNOTATION_LABELS", "FINEST_ACTIVITIES","MUSS_22_ACTIVITIES","MUSS_3_POSTURES","MUSS_6_ACTIVITY_GROUPS","MDCAS","RIAR_17_ACTIVITIES","SEDENTARY_AMBULATION_CYCLING","MUSS_22_ACTIVITY_ABBRS"
    ]
    exclude_labels = ['Unknown', 'Transition']
    dataset = dataset.loc[np.logical_not(dataset[train_target].
                                         isin(exclude_labels)), :]
    y = dataset[train_target].values
    indexed_dataset = dataset.set_index(index_cols)
    X = indexed_dataset.values
    feature_order = list(indexed_dataset.columns)
    if model_type == 'svm':
        selected_model = svm_model
    elif model_type == 'rf':
        selected_model = rf_model
    model, scaler, training_accuracy = selected_model(X, y)
    return model, scaler, training_accuracy, feature_order


def save_model(model_path, target, model, scaler, training_accuracy,
               feature_order, class_mapping):
    model_bundle = {
        'model_file': os.path.basename(model_path),
        'name': target,
        'model': model,
        'scaler': scaler,
        'training_accuracy': training_accuracy,
        'feature_order': feature_order,
        'class_mapping': class_mapping
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # main(input_folder='D:/data/muss_data/',
    #      debug=True, sites='DW', targets='MDCAS')
    run(main)
