from padar_converter.mhealth import dataset, fileio
import pandas as pd
import numpy as np


def include_pid(input_file):
    exceptions = pd.read_csv(dataset.find_pid_exceptions(input_file))
    pid = dataset.get_pid(input_file)
    if np.any(pid == exceptions['PID'].values):
        return False
    else:
        return True


def get_offset(input_file):
    offset_mapping_file = dataset.find_offset_mapping(input_file)
    pid = dataset.get_pid(input_file)
    if bool(offset_mapping_file):
        offset_mapping = fileio.load_offset_mapping(offset_mapping_file)
        offset_in_secs = float(offset_mapping.loc[offset_mapping['PID'] == pid, offset_mapping.columns[1]].values[0])
    else:
        offset_in_secs = 0
    return offset_in_secs
        

def get_orientation_correction(input_file):
    orientation_corrections_file = dataset.find_orientation_corrections(input_file)
    pid = dataset.get_pid(input_file)
    if bool(orientation_corrections_file):
        orientation_corrections = fileio.load_orientation_corrections(orientation_corrections_file)
        orientation_correction = orientation_corrections.loc[orientation_corrections['PID'] == pid, orientation_corrections.columns[3:6]]
        if orientation_correction.empty:
            orientation_correction = np.array(['x', 'y', 'z'])
        else:
            orientation_correction = orientation_correction.values[0]
    else:
        orientation_correction = np.array(['x', 'y', 'z'])
    return orientation_correction