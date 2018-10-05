import pandas as pd
import os
import numpy as np
from padar_converter.mhealth import dataset

def _merge(feature_set, class_set):
    joined_set = feature_set.merge(class_set, how='outer', on=['START_TIME', 'STOP_TIME', 'GROUP0'])
    joined_set = joined_set.dropna()
    return joined_set

def subset_and_merge(feature_set, class_set, sensor_placements, pids):
    feature_set = feature_set.loc[feature_set['GROUP0'].isin(pids),:]
    class_set = class_set.loc[class_set['GROUP0'].isin(pids),:]

    joined_set = feature_set.groupby(['GROUP0', 'GROUP2']).apply(_merge, class_set)
    joined_set = joined_set.loc[np.logical_not(joined_set['ACTIVITY'].isin(['Transition', 'Unknown'])),:]
    return joined_set

if __name__ == '__main__':
    input_folder = 'D:/data/spades_lab'
    output_folder = os.path.join(
        input_folder, 'DerivedCrossParticipants', 'location_matters')
    feature_set_file = os.path.join(output_folder, 'location_matters.feature.csv')
    class_set_file = os.path.join(output_folder, 'location_matters.class.csv')
    feature_set = pd.read_csv(feature_set_file, parse_dates=[0,1], infer_datetime_format=True)
    class_set = pd.read_csv(class_set_file, parse_dates=[0,1], infer_datetime_format=True)

    pids = dataset.get_pids(input_folder)
    sensor_placements = ['None']
    joined_set = subset_and_merge(feature_set, class_set, sensor_placements=sensor_placements, pids=pids)
    output_filepath = os.path.join(output_folder, '_'.join(pids)[:10] + '_' + '_'.join(sensor_placements) + '_train.csv')
    joined_set.to_csv(output_filepath, index=False)