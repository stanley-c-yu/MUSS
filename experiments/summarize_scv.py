import pandas as pd
import os
import numpy as np
from padar_converter.mhealth.dataset import find_pid_exceptions


def summarize_orientation_variations(location_mapping, orientation_corrections, exceptions):
    orientation_variations = location_mapping.merge(
        orientation_corrections, how='left')
    grouper = orientation_variations.columns[2]

    exception_list = exceptions['PID'].values
    orientation_variations = orientation_variations.loc[~orientation_variations['PID'].isin(
        exception_list), :]

    def summarize(group_df):
        percentage = (group_df['X'].size -
                      group_df['X'].isna().sum()) / group_df['X'].size
        result = pd.Series(
            data={'PERCENTAGE': percentage, 'TOTAL_NUMBER_OF_SENSORS': group_df.shape[0]})
        return result

    result = orientation_variations.groupby(
        grouper).apply(summarize)
    result = result.reset_index()
    return result


if __name__ == '__main__':
    input_folder = 'D:/data/spades_lab'
    location_mapping = pd.read_csv(os.path.join(
        input_folder, 'DerivedCrossParticipants', 'location_mapping.csv'))
    orientation_corrections = pd.read_csv(os.path.join(
        input_folder, 'DerivedCrossParticipants', 'orientation_corrections.csv'))
    exceptions = pd.read_csv(os.path.join(
        input_folder, 'DerivedCrossParticipants', 'pid_exceptions.csv'))
    result = summarize_orientation_variations(
        location_mapping, orientation_corrections, exceptions)
    output_file = os.path.join(
        input_folder, 'DerivedCrossParticipants', 'scv', 'summary_of_ov.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    result.to_csv(output_file)
