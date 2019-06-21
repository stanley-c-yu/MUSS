import os
import subprocess
from clize import run
import logging
from pack_data import pack_data

def _tagging(version, message=None):
    tagging_cmd = ['git', 'tag', '-a', version]
    if message is not None:
        tagging_cmd.append('-m')
        tagging_cmd.append('"' + message + '"')
    subprocess.run(' '.join(tagging_cmd), check=True, shell=True)

def create_dist(version, *, dataset=None, sample_result=None, message=None):
    logging.info('Tagging distribution...')
    _tagging(version, message=message)
    logging.info('Distribution is tagged with {}'.format(version))
    logging.info('Creating distribution repo...')
    dist_folder = os.path.join('dists', version)
    os.makedirs(dist_folder, exist_ok=True)
    logging.info('Distribution repo: {}'.format(dist_folder))
    if dataset is None:
        logging.warn('No dataset is provided, skipping packing data asset')
    else:
        logging.info('Packing data assets...')
        pack_data(dataset, dist_folder, 'muss_data.tar.gz')
        logging.info('Data asset is packed.')
    if sample_result is None:
        logging.warn('No sample result is provided, skipping packing result asset')
    else:
        logging.info('Packing result assests...')
        pack_data(sample_result, dist_folder, 'sample_reproduction_results.tar.gz')
        logging.info('Result asset is packed.')
    

if __name__ == "__main__":
    logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level, format='[%(levelname)s] %(asctime)-15s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    run(create_dist)