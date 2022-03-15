# README

## setup of the project

The project can be cloned from github.com with:
```
git clone https://github.com/dzhakparov/mlm-sos-all.git
```

### build new conda environment if cloned project
If this project is cloned to a new system, the environment can be built with:

    $ conda env create --file environment.yml


### update conda environment to environment.yaml
If new external packages are installed in the local conda environment 'environment.yml'-file have to be updated with:

    $  conda env export --name sosall > environment.yml

### manual installation of pandas-profiling
Some packages (for example own-written packages) have to be installed manually. Go to folder 'pkg' and 
install packages with the following instruction:

```
    $ pip install pandas-profiling
    $ pip install misc-0.0.11-py3-none-any.whl
    $ pip install PredictorPipeline-0.0.18-py3-none-any.whl
    $ pip install sklearn_custom-0.0.23-py3-none-any.whl
    $ pip install statTests-0.0.3-py3-none-any.whl
    $ pip install saxpy
```


## Information for PyCharm Users

If the paths to the files in folder 'src' and to config.py are not shown properly (red -> can't reference to them)

* **delete content root**: File -> settings -> Project: sosall -> Project structure
* **add content root**: File -> settings -> Project: sosall -> Project structure -> 
**+** '/home/schmidmarco/Documents/CODE/PROJECTS/sosall/data_preperation'


## Preprocessing Data

To be able to run jupyter-notebook 'global_preprocessing.ipynb' a folder with **data/input** (not included in 
clone!) has to be built. At least these 4 files have to be in this folder:

* raw_data_file defined in config (exp. 'SOSALL Master clinical data_NL_ms1_kb0.csv')
* specification_file defined in config (exp. 'description_data_ms8_kb2.csv')
* mapping_file defined in config (exp. 'mapping_metadatanames_pid_def.csv')
* meta-data_file defined in config (exp. 'metadata.txt')


If all files are available the notebook can be executed. A folder **'output'** and a subfolder with timestamp
and some other config-information is build. All results, plots and files are stored there.


## More Information
More Information is available under: 
* **'sosall/docs/build/html/index.html'**