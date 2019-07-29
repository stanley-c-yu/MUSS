from download_data import download_data
from prepare_class_set import prepare_class_set
from prepare_feature_set import prepare_feature_set
from prepare_validation_set import main as prepare_validation_set
from run_validation_experiments import main as run_validation_experiments
from train_and_save_model import main as train_and_save_model
from compute_metrics import main as compute_metrics
from publication_figures import main as get_figures
import logging
from helper.utils import generate_run_folder
from clize import run
import datetime

def build_riar(*, sites='DW,DA', model_type='svm', force_fresh_data=True, debug=False, parallel=False, profiling=False, run_ts='new'):
    logging_level = logging.DEBUG if debug else logging.INFO
    scheduler = 'processes' if parallel else 'sync'
    logging.basicConfig(level=logging_level, format='[%(levelname)s] %(asctime)-15s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Downloading data...')
    data_folderpath = download_data(force=force_fresh_data)
    logging.info('Data is downloaded and unzipped to {}'.format(data_folderpath))

    if run_ts is 'new':
        run_ts=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    output_folder = generate_run_folder(data_folderpath, debug=debug, run_ts=run_ts)

    logging.info('Preparing class set...')
    classset_path = prepare_class_set(data_folderpath, output_folder=output_folder, debug=debug, scheduler=scheduler, profiling=profiling, force=force_fresh_data)
    logging.info('Class set is generated to {}'.format(classset_path))
    logging.info('Preparing feature set...')
    feature_set_path = prepare_feature_set(data_folderpath, output_folder=output_folder, debug=debug, scheduler=scheduler, profiling=profiling, force=force_fresh_data)
    logging.info('Feature set is generated to {}'.format(feature_set_path))
    logging.info('Preparing validation set...')
    dataset_path = prepare_validation_set(data_folderpath, output_folder=output_folder, sites=sites, feature_types='MO', include_nonwear=False, debug=debug, scheduler=scheduler, profiling=profiling, force=force_fresh_data, target='RIAR_17_ACTIVITIES')
    logging.info('Validation set are saved to {}'.format(dataset_path))
    logging.info('Running validation experiments...')
    prediction_path = run_validation_experiments(data_folderpath, output_folder=output_folder, sites=sites, feature_set='MO', target='RIAR_17_ACTIVITIES', include_nonwear=False, model_type=model_type, debug=debug, scheduler=scheduler, profiling=profiling, force=force_fresh_data)
    logging.info('Prediction results are saved to {}'.format(prediction_path))
    logging.info('Train and save RIAR model...')
    model_folder = train_and_save_model(data_folderpath, output_folder=output_folder, debug=debug, targets='RIAR_17_ACTIVITIES', feature_set='MO', sites=sites, model_type=model_type, include_nonwear=False)
    logging.info('Model is saved to {}'.format(model_folder))


if __name__ == "__main__":
    build_riar(sites='DW,DA', model_type='svm', force_fresh_data=False, debug=False, parallel=True, profiling=False, run_ts='2019-07-11-15-46-02')