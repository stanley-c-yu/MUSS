import datetime
import os


def generate_run_folder(input_folder, debug=False):
    now = datetime.datetime.today()
    date_now = now.strftime('%Y-%m-%d')
    output_folder = os.path.join(input_folder, 'DerivedCrossParticipants',
                                 date_now, 'product_run')
    if debug:
        output_folder = os.path.join(input_folder, 'DerivedCrossParticipants',
                                     date_now, 'debug_run')
    return output_folder