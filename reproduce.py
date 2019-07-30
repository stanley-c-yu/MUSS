from download_data import download_data
from prepare_class_set import prepare_class_set
from prepare_feature_set import prepare_feature_set
from prepare_validation_set import main as prepare_validation_set
from run_validation_experiments import main as run_validation_experiments
from compute_metrics import main as compute_metrics
from publication_figures import main as get_figures
import logging
from helper.utils import generate_run_folder
from clize import run
import datetime


def reproduce(*, force_fresh_data=True, debug=False, parallel=False, profiling=False, run_ts='new', name=None, sampling_rate=80, resample_sr=80):
    logging_level = logging.DEBUG if debug else logging.INFO
    scheduler = 'processes' if parallel else 'sync'
    logging.basicConfig(level=logging_level, format='[%(levelname)s] %(asctime)-15s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Downloading data...')
    data_folderpath = download_data(force=force_fresh_data)
    logging.info('Data is downloaded and unzipped to {}'.format(data_folderpath))

    if run_ts is 'new':
        run_ts=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    output_folder = generate_run_folder(data_folderpath, debug=debug, run_ts=run_ts, name=name)

    logging.info('Preparing class set...')
    classset_path = prepare_class_set(data_folderpath, output_folder=output_folder, debug=debug, scheduler=scheduler, profiling=profiling, force=force_fresh_data)
    logging.info('Class set is generated to {}'.format(classset_path))
    
    logging.info('Preparing feature set...')
    feature_set_path = prepare_feature_set(data_folderpath, output_folder=output_folder, debug=debug, scheduler=scheduler, profiling=profiling, force=force_fresh_data, sampling_rate=sampling_rate, resample_sr=resample_sr)
    logging.info('Feature set is generated to {}'.format(feature_set_path))
    
    logging.info('Preparing validation sets...')
    dataset_path = prepare_validation_set(data_folderpath, output_folder=output_folder, debug=debug, scheduler=scheduler, profiling=profiling, force=force_fresh_data, include_nonwear=False)
    logging.info('Validation sets are saved to {}'.format(dataset_path))
   
    logging.info('Running validation experiments...')
    prediction_path = run_validation_experiments(data_folderpath, output_folder=output_folder, debug=debug, scheduler=scheduler, profiling=profiling, force=force_fresh_data, include_nonwear=False, model_type='svm')
    logging.info('Prediction results are saved to {}'.format(prediction_path))
  
    logging.info('Computing metrics...')
    metric_path, cm_path = compute_metrics(data_folderpath, output_folder=output_folder, debug=debug, scheduler=scheduler, profiling=profiling, force=force_fresh_data)
    logging.info('Metrics results are saved to {}'.format(metric_path))
    logging.info('Confusion matrices are saved to {}'.format(cm_path))
   
    logging.info('Generating publication figures and tables...')
    figure_path = get_figures(data_folderpath, output_folder=output_folder, debug=debug, force=force_fresh_data)
    logging.info('Figures and tables are saved to {}'.format(figure_path))

if __name__ == "__main__":
    # run(reproduce)
    reproduce(force_fresh_data=False, debug=True, parallel=True, profiling=False, run_ts='2019-07-15-16-42-19', sampling_rate=80, resample_sr=80)