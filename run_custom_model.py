import pickle
from clize import run
import os
import pandas as pd
import numpy as np
from helper.utils import generate_run_folder
import logging


def run_custom_model(input_folder,
                     *,
                     debug=False,
                     dataset_file,
                     model_file):
    """Run a saved model on a new dataset

    :param input_folder: Folder path of the input NEW raw dataset.
    :param debug: Use this flag to output results to 'debug_run' folder of the NEW raw dataset.
    :param dataset_file: the filename (not the full path) of a dataset file after running "prepare_validation_sets.py".
    :param model_file: the full path of the model file to test
    """
    output_folder = generate_run_folder(input_folder, debug=debug)
    result_folder = os.path.join(output_folder, 'tests')
    os.makedirs(result_folder, exist_ok=True)
    dataset_path = os.path.join(output_folder, 'datasets', dataset_file)
    feature_df = pd.read_csv(
        dataset_path, parse_dates=[0, 1], infer_datetime_format=True)
    feature_df = feature_df.dropna()
    indexed_feature_df = feature_df.set_index([
        'START_TIME', 'STOP_TIME', 'PID', 'SID', 'SENSOR_PLACEMENT',
        'FEATURE_TYPE'
    ])
    p_df = feature_df
    with open(model_file, 'rb') as mf:
        model_bundle = pickle.load(mf)
        feature_order = model_bundle['feature_order']
        ordered_df = indexed_feature_df.loc[:, feature_order]
        X = ordered_df.values
        name = model_bundle['name']
        class_labels = model_bundle['model'].classes_
        try:
            scaled_X = model_bundle['scaler'].transform(X)
            predicted_labels = model_bundle['model'].predict(scaled_X)
        except:
            predicted_labels = X.shape[0] * [np.nan]
        p_df['PREDICTION'] = predicted_labels
    result_path = os.path.join(
        result_folder,
        dataset_file.replace('dataset.csv',
                             name.lower() + '_prediction.csv'))
    p_df.to_csv(result_path, index=False)
    logging.info('Saved ' + result_path)


if __name__ == '__main__':
    # example
    # run_custom_model('./muss_data', model_file='./muss_data\DerivedCrossParticipants\product_run\models\DW_DA.MO.muss_3_postures_svm_model.pkl', dataset_file='DW_DA.MO.dataset.csv')
    run(run_custom_model)
