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

def _compress_data_with_python(source_dir, out_dir, out_name):
    def exclude(tarinfo):
        if 'DerivedCrossParticipants' in tarinfo.name:
            return None
        else:
            return tarinfo
    with tarfile.open(os.path.join(out_dir, out_name), "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir), filter=exclude)


def pack_data(source_dir, out_dir, out_name):
    os.makedirs(out_dir, exist_ok=True)
    logging.info('Use Python tar module to compress data...')
    _compress_data_with_python(source_dir, out_dir, out_name)
    logging.info('Compression completed')

if __name__ == "__main__":
    logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level, format='[%(levelname)s] %(asctime)-15s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    run(pack_data)