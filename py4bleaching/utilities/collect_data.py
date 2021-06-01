import os, re
import pandas as pd
import numpy as np
from loguru import logger
import glob
import shutil


def move_files(folders, file_types=False):
    if not file_types:
        file_types = {
        'hsp_colocalised': 'Hsp_colocal_traj', 
        'hsp_noncolocal': 'Hsp_non-colocal_traj', 
        'client_colocalised': 'Client_colocal_traj', 
        'client_noncolocal':'Client_non-colocal_traj'
        }
    for old_folder, (new_folder, filetype) in folders.items():
        old_files = [filename for filename in os.listdir(f'{input_folder}{old_folder}') if '.csv' in filename]
        if not os.path.exists(f'{output_folder}{new_folder}'):
            os.makedirs(f'{output_folder}{new_folder}')
        for x, filename in enumerate(old_files): 
            shutil.copyfile(f'{input_folder}{old_folder}{filename}', f'{output_folder}{new_folder}{file_types[filetype]}{x}.csv')


if __name__ == "__main__":
   
    #change these for experiment
    input_folder = 'D:/Experiment 17/TIRF/'
    output_folder = 'CAITLIN PROTOCOL example data/20210113_Exp_17_647data/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #change these for the experiment you're using
    folders = {
    #  '20201209_110525_171experiment 14 aB x linked at 5uM/Trajectories/':('crosslinked-aBc-5uM/non-coloc/', hsp_noncolocal), 

    #  '20201209_125020_554exp 14 xlinked at 20 uM/Trajectories/':('crosslinked-aBc-20uM/non-coloc/', hsp_noncolocal), 

    # 'non-colocalised aB-c/': ('noncrosslinked-aBc/non-coloc/', hsp_noncolocal,),
    # 'colocalised aB-c/': ('noncrosslinked-aBc/coloc/', hsp_colocalised,),

    'CLIC UNHEATED 13012021/Trajectories/':('CLIC_not-heated/non-coloc/', 'client_noncolocal'), 
    'CLIC HEATED 13012021/Trajectories/':('CLIC_heated/non-coloc/', 'client_noncolocal')
        
        }

    move_files(folders)



