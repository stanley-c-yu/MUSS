import datetime
import os


def generate_run_folder(input_folder, debug=False, name=None):
    output_folder = os.path.join(
        input_folder, 'DerivedCrossParticipants', 'product_run')
    if debug:
        output_folder = os.path.join(
            input_folder, 'DerivedCrossParticipants', 'debug_run')
    if name is not None:
        output_folder += '_' + name
    return output_folder


def strip_path(path):
    if path.endswith('/'):
        return path[:-1]


def print_args(args):
    for name, arg in args.items():
        print(name + ': ' + str(arg))
