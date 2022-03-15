# starting point of SOSALL: preprocessing

import os
import time
import config
import warnings
from importlib import reload
from src.helpers import set_up_infrastructure, build_working_directory, copy_config
from src import statistical_tests, eda

if __name__ == '__main__':

    reload(config)  # reloads config every time and does not take it from cache (changes)
    warnings.filterwarnings("ignore")  # ignore warnings when training models
    start = time.time()
    output_folder = set_up_infrastructure(dt_string=config.dt_now) # building directory to store calculations
    copy_config(output_folder=output_folder)

    print(f" \n**** Running SOSALL **** \n\ntarget: {config.target}, \ngroup: {config.subset}\n"
          f"folder: {build_working_directory(dt_string=config.dt_now)}\n")

    os.system(" jupyter nbconvert --to notebook --inplace --execute src/global_preprocessing.ipynb")

    statistical_tests.run(output_folder=output_folder)  # applies statistical tests to defined subset
    eda.run(output_folder=output_folder)  # applies explorative data-analysis to defined subset
    end = time.time()
    print(f"calculation finished! used time: {round((end - start) / 60, 2)} min.")
