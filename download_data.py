import wget
import tarfile
import os
from clize import run
import subprocess
import logging

def check_tar_available():
    tar_info = subprocess.getoutput('tar')
    if 'tar: Must specify one of -c, -r, -t, -u, -x' == tar_info:
        return True
    else:
        return False

def download_data(*, keep_compressed=False, force=True):
    if os.path.exists(os.path.abspath('./muss_data/')) and not force:
        logging.info('Data has already existed, use the existing data')
        return os.path.abspath('./muss_data/')
    url = 'https://github.com/qutang/MUSS/releases/download/data/muss_data.tar.gz'
    filename = wget.download(url)
    if check_tar_available():
        logging.info('Using system tar command to decompress data file')
        decompress_cmd = ['tar', '-xzf', filename]
        subprocess.run(' '.join(decompress_cmd), check=True, shell=True)
    else:
        logging.info('Using Python tar module to decompress data file')
        tar = tarfile.open(filename)
        tar.extractall()
        tar.close()
    logging.info('Decompression completed')
    if not keep_compressed:
        os.remove(filename)
    return os.path.abspath('./muss_data/')

if __name__ == "__main__":
    logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level, format='[%(levelname)s] %(asctime)-15s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    run(download_data)