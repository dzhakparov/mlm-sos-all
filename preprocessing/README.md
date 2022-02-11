# README

This jupyter-notebook have been build from former own project 'SOSALL_data_preparation' (not used or old code has been 
removed, only last version of notebook stays as actual working file)

Goal if this 'preprocessing.ipynb' is to transform original raw data into preprocessed data that are usable for 
machine-learning and other analysis.

**Note:** No data have been imputed or something like that. Only transformations like renaming a.s.o. have been applied

## Preprocessing Data

To be able running **preprocessing_x.ipynb** you must have a folder **'data/modified_input'** with a couple of data:

* description_data_convert_columns_ms2.csv
* description_data_ms7_kb2.csv
* description_data_recoding_ms3.csv
* mapping_metadatanames_pid_def.csv
* formated_questionnaire_switzerland.xlsx
* metadata.txt
* SOSALL Master clinical data_NL_ms1_kb0.csv

The ending (index) of these files can be slightly different (new versions of these files), then this has to be adjusted 
in the upper part of **preprocessing_x.ipynb** (Read-in Raw-data -> specifications)

If all files are available the notebook can be executed. A folder **'output'** will be build and all files 
(pre-steps and plots) are stored there. This folder will always be cleared before every new run!

Every run also builds a folder **'logs'** that contains a file with thew according timestamp of evaluation. In these 
log-files one can see what happened in the according preprocessing step.

## update conda environment to environment.yaml
If new external packages are installed in the local conda environment 'environment.yml'-file have to be updated with:

    $  conda env export --name sosall > environment.yml

## build new conda environment if cloned project
If this project is cloned to a new system, the environment can be built with:

    $ conda env create --file environment.yml

or if the conda environment already exists, update this with:

    $ conda env update --file environment.yml --prune

## manual installation of pandas-profiling
```
    $ pip install pandas-profiling
```

## info for PyCharm user

If the paths to the files in folder 'src' and to config.py are not shown properly (red -> can't reference to them)

* **delete content root**: File -> settings -> Project: sosall -> Project structure
* **add content root**: File -> settings -> Project: sosall -> Project structure -> 
**+** '/home/schmidmarco/Documents/CODE/PROJECTS/sosall/data_preperation'