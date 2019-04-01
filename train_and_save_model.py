import glob
import pandas as pd
from helper.svm_model import svm_model
from helper.utils import generate_run_folder
import os
import pickle
from clize import run


def main(input_folder, debug, feature_set, *sites):
    run_folder = generate_run_folder(input_folder, debug=debug)
    dataset_folder = os.path.join(run_folder, 'datasets')
    sites = list(sites)
    train_and_save_model(dataset_folder, sites=sites, feature_set=feature_set)


def train_and_save_model(dataset_folder, sites=['DW', 'DA'], feature_set='MO'):
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
        targets = [
            'POSTURE', 'ACTIVITY', 'CLASSIC_SEVEN_ACTIVITIES',
            'SEDENTARY_AMBULATION_CYCLING'
        ]
        for target in targets:
            model_path = os.path.join(
                model_folder,
                os.path.basename(selected_file).replace(
                    'dataset',
                    target.lower() + '_model').replace('csv', 'pkl'))
            model, scaler, training_accuracy, feature_order = train_model(
                dataset, target)
            save_model(model_path, target, model, scaler, training_accuracy,
                       feature_order)


def train_model(dataset, target):
    index_cols = [
        "START_TIME", "STOP_TIME", "PID", "SID", "SENSOR_PLACEMENT",
        "FEATURE_TYPE", "ANNOTATOR", "ANNOTATION_LABELS", "ACTIVITY",
        "POSTURE", "ACTIVITY_GROUP", "CLASSIC_SEVEN_ACTIVITIES",
        "SEDENTARY_AMBULATION_CYCLING", 'ACTIVITY_ABBR'
    ]
    placements = dataset['SENSOR_PLACEMENT'].values[0]
    feature_type = dataset['FEATURE_TYPE'].values[0]
    y = dataset[target].values
    indexed_dataset = dataset.set_index(index_cols)
    X = indexed_dataset.values
    feature_order = list(indexed_dataset.columns)
    model, scaler, training_accuracy = svm_model(X, y)
    return model, scaler, training_accuracy, feature_order


def save_model(model_path, target, model, scaler, training_accuracy,
               feature_order):
    model_bundle = {
        'model_file': os.path.basename(model_path),
        'name': target,
        'model': model,
        'scaler': scaler,
        'training_accuracy': training_accuracy,
        'feature_order': feature_order
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    # run(main)
    main('D:/data/muss_data/', True, 'MO', 'DW', 'DA')
