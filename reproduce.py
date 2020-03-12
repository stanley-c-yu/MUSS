import datetime
import logging

from clize import run

import steps
from helper.utils import generate_run_folder


def reproduce(*, force_fresh_data=True, debug=False, parallel=False, profiling=False, name=None, sampling_rate=80, resample_sr=80):
    logging_level = logging.DEBUG if debug else logging.INFO
    scheduler = 'processes' if parallel else 'sync'
    logging.basicConfig(
        level=logging_level, format='[%(levelname)s] %(asctime)-15s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Downloading data...')
    data_folderpath = steps.download_data(force=force_fresh_data)
    logging.info(
        'Data is downloaded and unzipped to {}'.format(data_folderpath))

    output_folder = generate_run_folder(
        data_folderpath, debug=debug, name=name)

    logging.info('Preparing class set...')
    classset_path = steps.prepare_class_set(data_folderpath, output_folder=output_folder,
                                            debug=debug, scheduler=scheduler, profiling=profiling, force=force_fresh_data)
    logging.info('Class set is generated to {}'.format(classset_path))

    logging.info('Preparing feature set...')
    feature_set_path = steps.prepare_feature_set(data_folderpath, output_folder=output_folder, debug=debug, scheduler=scheduler,
                                                 profiling=profiling, force=force_fresh_data, sampling_rate=sampling_rate, resample_sr=resample_sr)
    logging.info('Feature set is generated to {}'.format(feature_set_path))

    logging.info('Preparing validation sets...')
    dataset_path = steps.prepare_validation_set(data_folderpath, output_folder=output_folder, debug=debug,
                                                scheduler=scheduler, profiling=profiling, force=force_fresh_data, include_nonwear=False)
    logging.info('Validation sets are saved to {}'.format(dataset_path))

    logging.info('Running validation experiments...')
    prediction_path = steps.run_validation_experiments(data_folderpath, output_folder=output_folder, debug=debug,
                                                       scheduler=scheduler, profiling=profiling, force=force_fresh_data, include_nonwear=False, model_type='svm')
    logging.info('Prediction results are saved to {}'.format(prediction_path))

    logging.info('Computing metrics...')
    metric_path, cm_path = steps.compute_metrics(
        data_folderpath, output_folder=output_folder,
        debug=debug, scheduler=scheduler, profiling=profiling, force=force_fresh_data)
    logging.info('Metrics results are saved to {}'.format(metric_path))
    logging.info('Confusion matrices are saved to {}'.format(cm_path))

    logging.info('Generating publication figures and tables...')
    figure_path = steps.get_figures(
        data_folderpath, output_folder=output_folder, debug=debug, force=force_fresh_data)
    logging.info('Figures and tables are saved to {}'.format(figure_path))


if __name__ == "__main__":
    run(reproduce)
    # reproduce(force_fresh_data=False, debug=True, parallel=True, profiling=False, run_ts='2019-07-15-16-42-19', sampling_rate=80, resample_sr=80)
