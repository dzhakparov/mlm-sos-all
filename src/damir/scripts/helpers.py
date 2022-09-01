import pandas as pd
def df_report(df):
    ### make a table with feature value counts, number of unique values and missing value counts
    counts = [] # list with value counts for every column
    feature_name = [] # list of feature names
    na_count = [] # list with na counts in every column
    perc_nan = [] # list with percentages of nas in every column
    length = [] # list of amount of unique values in every column
    for column in df.columns:
        ### calculate the value counts in each column
        val_counts = df[column].value_counts().to_dict()
        ### calculate the nan percentage in every column
        nan_sum = df[column].isna().sum()
        percent_missing = df[column].isnull().sum() * 100 / len(df[column])
        ### calculate the number of unique values
        length.append(len(df[column].unique()))
        ### append the values to list to construct the dataframe
        counts.append(val_counts)
        feature_name.append(column)
        na_count.append(nan_sum)
        perc_nan.append(percent_missing)

    ### construct dataframe
    ### for some reason counts NaN as a uniqe value but doesn't show it in the dictionary
    report = pd.DataFrame(list(zip(feature_name, counts, length, na_count, perc_nan)),
                          columns = ['variable','value_counts','unique_values','nan_counts','nan_percentage'])
    return report

# -*- coding: utf-8 -*-
import logging
from datetime import datetime, date
from pathlib import Path

"ALl credits for the script go to Vasily Tolkachev, CK-CARE"
def setup_logger(path_file, path_project, type = 'data'):

    path_file = Path(path_file)
    path_project = Path(path_project)

    filename = path_project / f'{type}_logs' / f'{path_file.name}_{date.today()}.log'
    (path_project / f'{type}_logs').mkdir(parents=True, exist_ok=True)

    # logging.basicConfig(filename = filename, filemode = 'w')

    logger = logging.getLogger()
    log_formatter = logging.Formatter('%(asctime)s %(message)s. ')

    file_handler = logging.FileHandler(filename, mode = 'w', encoding = 'utf-8')
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)

    # to output to console and write to log file simulteneously
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

    # all more important levels will be tracked as well
    logger.setLevel(logging.DEBUG)
    return logger