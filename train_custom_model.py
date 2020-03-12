import glob
import logging
import os
import pickle

import numpy as np
import pandas as pd
from clize import run

import steps
from helper.svm_model import rf_model, svm_model
from helper.utils import generate_run_folder


def main(input_folder,
         *,
         output_folder=None,
         debug=False,
         sr=80,
         targets='MUSS_3_POSTURES,MUSS_22_ACTIVITIES',
         feature_set='MO',
         sites='DW,DA', include_nonwear=False, model_type='svm', profiling=False):
    """Train and save a model given a arbitrary dataset stored in mhealth format.

    :param input_folder: Folder path of input raw dataset.
    :param output_folder: Auto path if None.
    :param debug: Use this flag to output results to 'debug_run' folder.
    :param targets: The list of groups of class labels, separated by ','.

                Allowed targets:
                'MUSS_3_POSTURES': includes "sitting", "upright", and "lying"
                'MUSS_22_ACTIVITIES': includes 22 daily activities used in the muss dataset
                'MDCAS', 'RIAR_17_ACTIVITIES'

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
    logging_level = logging.DEBUG if debug else logging.INFO
    scheduler = 'processes'
    force_fresh_data = False
    logging.basicConfig(
        level=logging_level, format='[%(levelname)s] %(asctime)-15s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    output_folder = generate_run_folder(
        input_folder, debug=debug)

    logging.info('Preparing class set...')
    classset_path = steps.prepare_class_set(
        input_folder, output_folder=output_folder,
        debug=debug, scheduler=scheduler, profiling=profiling, force=force_fresh_data)
    logging.info('Class set is generated to {}'.format(classset_path))

    logging.info('Preparing feature set...')
    feature_set_path = steps.prepare_feature_set(
        input_folder, output_folder=output_folder, debug=debug, scheduler=scheduler,
        profiling=profiling, force=force_fresh_data, sampling_rate=sr, resample_sr=sr)
    logging.info('Feature set is generated to {}'.format(feature_set_path))

    logging.info('Preparing validation sets...')
    dataset_path = steps.prepare_validation_set(
        input_folder, output_folder=output_folder, debug=debug,
        scheduler=scheduler, profiling=profiling, force=force_fresh_data, include_nonwear=include_nonwear)
    logging.info('Validation sets are saved to {}'.format(dataset_path))

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
        logging.info('Train ' + model_type + ' model for ' +
                     target + ' using ' + feature_set + ' feature set.')
        dataset_folder = os.path.join(
            output_folder, 'datasets' + suffix)
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
                target.lower() + '_' + model_type + '_model').replace('csv', 'pkl'))

        model, scaler, training_accuracy, feature_order = train_model(
            dataset, get_train_target(target), model_type=model_type)
        save_model(model_path, target, model, scaler, training_accuracy,
                   feature_order, class_mapping)
        logging.info('Save model to ' + model_path)


def get_train_target(target):
    if target == 'MUSS_22_ACTIVITIES' or target == 'MUSS_3_POSTURES' or target == 'MDCAS' or target == 'RIAR_17_ACTIVITIES':
        return target
    else:
        return 'MUSS_22_ACTIVITIES'


def train_model(dataset, train_target, model_type='svm'):
    index_cols = [
        "START_TIME", "STOP_TIME", "PID", "SID", "SENSOR_PLACEMENT",
        "FEATURE_TYPE", "ANNOTATOR", "ANNOTATION_LABELS", "FINEST_ACTIVITIES", "MUSS_22_ACTIVITIES", "MUSS_3_POSTURES", "MUSS_6_ACTIVITY_GROUPS", "MDCAS", "RIAR_17_ACTIVITIES", "SEDENTARY_AMBULATION_CYCLING", "MUSS_22_ACTIVITY_ABBRS"
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
    run(main)
    # examples
    # train mdcas model
    # main('./muss_data',
    #      output_folder=None,
    #      debug=False,
    #      sr=80,
    #      targets='MDCAS',
    #      feature_set='MO',
    #      sites='DW',
    #      include_nonwear=True,
    #      model_type='svm',
    #      profiling=False)
    # train model for active training
    # main('./muss_data',
    #      output_folder=None,
    #      debug=False,
    #      sr=80,
    #      targets='RIAR_17_ACTIVITIES',
    #      feature_set='MO',
    #      sites='DW,DA,DT',
    #      include_nonwear=False,
    #      model_type='svm',
    #      profiling=False
    #      )
