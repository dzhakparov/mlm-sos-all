import os
import pickle
import shutil
import socket
import config
import pandas as pd
from loguru import logger
from pprint import pformat
from config import loging_config
from config import ml_models_path as paths
from PredictorPipeline.predicting.predictor import Predictor


def run():

    logger.configure(**loging_config)
    logger.info(f"start initializing '{__name__}' on '{socket.gethostname().upper()}'!")

    data = _read_data()
    _build_store_folder()
    data = _build_subset(data)
    data = _build_target(data)
    data = _build_feature_columns(data)
    _scan_data(data)
    test_data, train_data = _build_train_test_data(data)

    logger.info(f"parameters: \n{pformat(config.parameters)}")
    logger.info(f"scoring-function: \n{pformat(config.scoring)}")
    logger.info(f"pipeline: \n{pformat(config.pipe)}")
    logger.info(f"cross-validation: \n{pformat(config.cv)}")

    # initiate Predictor object
    predictor = Predictor(train_data=train_data,
                          target=config.target,
                          test_data=test_data,
                          parameters=config.parameters,
                          pipeline=config.pipe,
                          cv=config.cv)

    predictor.fit(scoring=config.scoring,
                  refit=config.refit,
                  return_train_score=config.return_train_score,
                  cache=True,
                  cache_path=f"{paths['output_folder']}/predictor.pkl",
                  n_jobs=-1)

    logger.info(f"predictions successfully finished! Stored calculated object 'predictor.pkl' in "
                f"{paths['output_folder']}/predictor.pkl")


# *********************************** HELPER-FUNCTIONS ***************************************************************

def _build_train_test_data(data):

    # build train and test set (if necessary -> subset == 'both')
    if config.subset == "both":
        train_data = data[data.loc[:, 'country'] == 'South Africa']
        test_data = data[data.loc[:, 'country'] == 'Switzerland']
        train_data = train_data.drop(['country'], axis=1)
        test_data = test_data.drop(['country'], axis=1)
    else:
        train_data = data
        train_data = train_data.drop(['country'], axis=1)
        test_data = None

    logger.info(f"build train- and test-set: no obs. train = {train_data.shape[0]}, "
                f"no obse. test = {test_data.shape[0] if test_data is not None else 0}")
    logger.info(f"removed columns 'country' from train- and test-data")

    logger.info(f"train data [{train_data.shape}]:\n"
                f"\t{train_data.head(5)}\n")
    logger.info(f"test data:\n"
                f"\t{test_data.head(5) if test_data is not None else None}\n")

    _store_data(train_data, test_data)

    return test_data, train_data


def _build_feature_columns(data):
    # take all usable columns (inclusive 'country' -> any case)
    original_columns = data.columns
    if len(config.include_exclude_columns) != 0:
        if config.include:
            cols = config.include_exclude_columns
            inc = list(set([i for i in cols if i in data.columns] + ['country']))
            if config.target not in inc:
                inc = [config.target] + inc
            data = data.loc[:, inc]
        else:  # exclude
            cols = list(set([i for i in data.columns if i not in config.include_exclude_columns] + ['country']))
            if config.target not in cols:
                cols = [config.target] + cols
            data = data.loc[:, cols]
    excluded_columns = [i for i in original_columns if i not in data.columns]
    logger.info(f"Columns used for modelling [{len(data.columns) - 1}]: \n{pformat(sorted(list(data.columns)[1:]))}")
    logger.info(f"excluded columns are [{len(excluded_columns)}]: \n{pformat(sorted(list(excluded_columns)))}")
    return data


def _build_target(data):
    if config.target == 'diagnosis':
        labels = {'HC': 0, 'AD': 1}
        data.replace({"diagnosis": labels}, inplace=True)
        data['diagnosis'] = data['diagnosis'].astype("int")
        data.drop(['diagnosis_location'], axis=1, inplace=True)
        logger.info(f"recoded 'diagnosis' with {labels} and dropped column 'diagnosis_location'")
    else:
        labels = {'AD_Urban': 0, 'AD_Rural': 1, 'HC_Urban': 2, 'HC_Rural': 3}
        data.replace({"diagnosis_location": labels}, inplace=True)
        data['diagnosis_location'] = data['diagnosis_location'].astype("int")
        data.drop(['diagnosis'], axis=1, inplace=True)
        logger.info(f"recoded 'diagnosis_location' with {labels} "
                    f"and dropped column 'diagnosis'")

    with open(f"{paths['output_folder']}/labels.pkl", 'wb') as handle:
        pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"Chose '{config.target}' as target!")
    return data


def _build_subset(data):
    if config.subset != "both":
        data = data[data['country'] == config.subset]
        cols = data.columns
        data = data.dropna(how='all', axis=1)  # drops empty columns
        if len(data.columns) != len(cols):
            logger.info(f"dropped empty columns: {[i for i in cols if i not in data.columns]}")
        logger.info(f"Chose '{config.subset}' as country selection!")
    return data


def _build_store_folder():
    # create folder to store results
    try:
        os.mkdir(paths['output_folder'])
        logger.info(f"Directory '{paths['output_folder']}' created!")
    except FileExistsError:
        for filename in os.listdir(paths['output_folder']):
            file_path = os.path.join(paths['output_folder'], filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error('Failed to delete %s. Reason: %s' % (file_path, e))
        logger.info(f"Directory '{paths['output_folder']}' successfully cleared!")


def _read_data():
    # read in data
    try:
        with open(paths['input_file'], 'rb') as infile:
            data = pickle.load(infile)
            logger.info(f"File '{paths['input_file']}' successfully imported!")
    except (FileNotFoundError, Exception):
        logger.error(
            f"Something went wrong with opening '{paths['input_file']}'. Please watch paths in config.py and check if "
            f"file exists in according folder!")
    return data


def _store_data(train_data, test_data):
    data = [(train_data, "train_data"), (test_data, "test_data")]
    for item, name in data:
        if item is not None:
            item.to_csv(f"{paths['output_folder']}/{name}.csv", index=True, header=True)
            item.to_pickle(f"{paths['output_folder']}/{name}.pkl")
    logger.info(f"stored train- and test-data successfully as .csv and .pkl in '{paths['output_folder']}'!")


def _scan_data(data):
    """ scans data for inconsistencies """

    pd.set_option('display.max_rows', 500)
    types = data.dtypes
    nas = data.isnull().sum(axis=0)
    out = pd.concat([types, nas], axis=1)
    out.columns = ['types', 'missing values']
    logger.info(f"data: \n{out}")


if __name__ == '__main__':
    run()
