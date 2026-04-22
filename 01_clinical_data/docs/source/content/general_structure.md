# File structure and Code explanation

The preprocessing part have been done by Damir Zhakparov. Result is a file 
```DataTransformed.txt``` in folder ```data/input```. From this file a .csv version 
has been calculated as well.

There are 5 main-files to be run in the following order (because the output if a certain
script/notebook can be an input of the next one):

1. file_transformation.ipynb
2. statistical_test.ipynb
3. build_ml_models.ipynb
4. evaluate_ml_models.ipynb
5. explainer_dashboard.py

## file_transformation.ipynb
```DataTransformed.csv``` will be transformed to ```preprocessed_data.pkl``` (folder: data/output/)
* change of column-types (categorical)

## statistical_test.ipynb
* reads-in ```preprocessed_data.pkl``` and performs statistical analysis on each feature.
* The statistical test will be applied on 3 different groups: 'diagnosis', 'location', 'diagnosis_location'
* Output-files are ```statistics_diagnosis.csv```, ```statistics_diagnosis_location.csv```, 
  ```statistics_location.csv``` in the specified output-folder

## build_ml_models.ipynb
## evaluate_ml_models.ipynb
## explainer_dashboard.py