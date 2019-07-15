import datetime
import os


def generate_run_folder(input_folder, debug=False, run_ts=None, name=None):
    now = datetime.datetime.today()
    date_now = now.strftime('%Y-%m-%d')
    output_folder = os.path.join(input_folder, 'DerivedCrossParticipants', 'product_run')
    if debug:
        output_folder = os.path.join(input_folder, 'DerivedCrossParticipants', 'debug_run')
    if name is not None:
        output_folder += '_' + name
    if run_ts is not None:
        output_folder += '_' + run_ts
    return output_folder


def strip_path(path):
    if path.endswith('/'):
        return path[:-1]


def print_args(args):
    for name, arg in args.items():
        print(name + ': ' + str(arg))