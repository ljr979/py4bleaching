import gzip
import os
import re
import shutil
import urllib.request as request
from contextlib import closing

import requests
from loguru import logger

logger.info('Import OK')


def download_model(model_name, output_folder, URL='https://github.com/ljr979/py4bleaching_models/raw/main/'):
    """
    Worker function to download and save file from URL.
    
    inputs
    ======
    model_name: (str) name of folder to download model from (including extension)
    url: (str) complete location of file to be downloaded
    output_path: (str) relative or complete path to directory where folder will be saved.
    returns:
    ======
    None
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        with closing(request.urlopen(f'{URL}{model_name}/model.hdf5')) as r:
            with open(f'{output_folder}{model_name}.hdf5', 'wb') as f:
                shutil.copyfileobj(r, f)
        logger.info(f'Downloaded {model_name}')
    except:
        logger.info(f'Downloaded failed for {model_name}.')


if __name__ == "__main__":
#Find the info for which model to dowload in each experiment at the https://github.com/ljr979/py4bleaching_models repository, then add the details into the parameters (adjust output folder to be the experiment repository)
    output_folder = 'model_for_prediction/'
    download_model(model_name='Model_1', output_folder=output_folder)