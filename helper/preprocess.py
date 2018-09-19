from padar_converter.mhealth import dataset, fileio
import pandas as pd
import numpy as np


def exclude_pid(input_file):
    exceptions = pd.read_csv(dataset.find_pid_exceptions(input_file))
    pid = dataset.get_pid(input_file)
    if pid in exceptions['PID']:
        return True
    else:
        return False


def get_offset(input_file):
    offset_mapping_file = dataset.find_offset_mapping(input_file)
    pid = dataset.get_pid(input_file)
    if bool(offset_mapping_file):
        offset_mapping = fileio.load_offset_mapping(input_file)
        offset_in_secs = offset_mapping.iloc[offset_mapping['PID'] == pid, 1]
    else:
        offset_in_secs = 0
    return offset_in_secs
        

def get_orientation_correction(input_file):
    orientation_corrections_file = dataset.find_orientation_corrections(input_file)
    pid = dataset.get_pid(input_file)
    if bool(orientation_corrections_file):
        orientation_corrections = fileio.load_orientation_corrections(input_file)
        orientation_correction = orientation_corrections.iloc[orientation_corrections['PID'] == pid, 3:6].values
    else:
        orientation_correction = np.array(['x', 'y', 'z'])
    return orientation_correction