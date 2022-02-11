import pandas as pd
import numpy as np
import logging
import errno
import os
import sys
from datetime import datetime
import missingno as msno
import matplotlib.pyplot as plt

# TODO: df_specifications not in init but in methods (rename,...)
# TODO: rename: dictionary
# TODO: write method 'remain_columns'
# TODO: config.yml (base configuration in folder) and possibility to pass confog.yml to init (overwrites duplicates items)


class DataFrameConverter:
    """ DataFrameConverter converts columns in a pandas DataFrame.

    :param df_data: pandas DataFrame with raw data
    """

    def __init__(self, df_data):
        """ initializes class DataFrameConverter

        :param df_data: pandas DataFrame with raw data
        """
        self._initial_data = self.processed_data = df_data

        # TODO: create .yaml file for config (inlk. logger)
        self._format_logging = [3, 10, 20, 30]  # format sizes for time(milliseconds), level, module and function
        self._length_logging_messages = 90  # 100000 for 'normal' format (all in one row)

        self._mydir = os.path.join(os.getcwd(), 'logs' ,datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self._create_folder_for_files()  # create folder to store processed data

        # basic configuration of logger
        logging.basicConfig(
            filename=f'{self._mydir}/summary.log',
            filemode='w',
            level=logging.INFO,
            format=f'%(asctime)s.%(msecs)-{self._format_logging[0]}d %(levelname)-{self._format_logging[1]}s %(module)-'
                   f'{self._format_logging[2]}s - %(funcName)-{self._format_logging[3]}s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        self.logger = logging.getLogger()
        self.logger.info("initialize class 'DataFrameConverter'...")

        self._store_initial_data()  # stores initial .csv-file
        self._add_initial_file_infos()  # describes input file (shape,...)

        logging.info("class 'DataFrameConverter' successfully initialized")

    def _create_folder_for_files(self):
        """ creates a folder in the current working directory (timestamp) for saving all outputs and logs.
        """
        try:
            os.makedirs(self._mydir)
            # logging.info(f"\n  - folder [{self._mydir}] for storing files successfully created")
        except OSError as e:
            if e.errno != errno.EEXIST:
                # logging.error("\n  - folder for storing files could not be created!")
                raise  # This was not a "directory exist" error..

    def _store_initial_data(self):
        """ stores initial pandas DataFrame as .csv-file.
        """
        self._initial_data.to_csv(f"{self._mydir}/initial_data.csv", index=False)
        logging.info(f"\n  - initial_data.csv successfully stored in {self._mydir}")

    def _add_initial_file_infos(self):
        """ logging initial shape of data.
        """
        shape = self.processed_data.shape
        names = list(self.processed_data.columns)
        logging.info(f"\n  - initial shape: {shape}")
        logging.info(f"\n  - initial column names: {self._format_logging_output(text=names, length=self._length_logging_messages)}")
        # logging.info(f"    initial column names: {names}")

    def rename(self, names=None):
        """ renames header of DataFrame with help of a two column DataFrame or dictionary.

        :param names: pandas DataFrame with original names in first column and new names in second column OR dictionary with column name to be changed as key and new name as value.
        """

        logging.info("rename() called")

        if names is not None:
            original_names = list(self.processed_data.columns)

            # convert DataFrame to dictionary
            if isinstance(names, pd.DataFrame):

                len_names_before_transformation = len(names)
                dups = names.iloc[:,0].duplicated()
                transformed_names = dict(zip(names.iloc[:,0].values, names.iloc[:,1].values))
                len_names_after_transformation = len(transformed_names)

                # existing names may not be duplicated
                if len_names_after_transformation != len_names_before_transformation:
                    logging.info(f"\n  - duplicate names in data: {self._format_logging_output(names[dups], 100000)}")
                    logging.error(f"\n  - Multiple equal entries in first column! "
                                  f"This will cause problems in renaming! Please change naming in raw_data first!\n"
                                  f"  #### PROGRAM EXIT #####")
                    sys.exit()

                else:
                    names = transformed_names

            logging.info(f"\n  - names [{len(names)}] = {self._format_logging_output(names, length=self._length_logging_messages)}")

            successful_renamings = []
            unsuccessful_renamings = []

            # loop over each item in dictionary
            for key, value in names.items():
                names = dict([(key, value)])
                if key in original_names:
                    self.processed_data.rename(columns=names, inplace=True)
                    successful_renamings.append(names)
                else:
                    unsuccessful_renamings.append(key)

            logging.info(f"\n  - successful renamings [{len(successful_renamings)}]: "
                         f"{self._format_logging_output(successful_renamings, length=self._length_logging_messages)}")
            if len(unsuccessful_renamings) > 0:
                logging.warning(f"\n  - missed renamings, no valid headers [{len(unsuccessful_renamings)}]: "
                                f"{self._format_logging_output(unsuccessful_renamings, length=self._length_logging_messages)}")

        else:
            logging.warning("\n  - failed: A specification-file has to defined!")

    def convert_columns(self, type=None):
        """converts columns according to its specification in type.

        :param type: pandas DataFrame with column name in the first column and desired object type (object, category, int64, float64,...) in the second column. If there is a third column (optional) with recoding-input this values will be taken as dafault categories. If there is no third column only the existing items from data will be taken as categories.
        """

        logging.info("convert_columns() called")
        cols = self.processed_data.columns
        not_in_processed_data = list(np.setdiff1d(list(type.iloc[:, 0]), list(cols)))

        if len(not_in_processed_data) > 0:
            logging.warning(f"\n  - there have been items in list which are not in data[{len(not_in_processed_data)}]: "
                            f"{self._format_logging_output(text=not_in_processed_data, length=self._length_logging_messages)}")

        if type is not None:

            existing_categories = {}

            # adds existing categories to dictionary with column name as key and existing categories as values (exp.
            # {gender: ['m', 'w']}
            if type.shape[1] == 3:  # 3 columns
                # build dict from column 3
                data = type[type.iloc[:, 1] == 'category']
                notnull = pd.notnull(data.iloc[:, 2])  # 'data.iloc[:, 2]' equals 'data.categories'
                data = data[notnull]

                for row in data.iterrows():
                    result = [x.strip() for x in row[1][2].split(',')]
                    existing_categories.update({row[1][0]: result})

                logging.info(f"\n   - wished categories: {existing_categories}")

            missing_columns = []

            # iterate each column in processed_data
            for idx, item in enumerate(cols):
                # exp: item: 'Name'

                if item not in type.iloc[:, 0].values:
                    missing_columns.append(item)
                    # logging.warning(f"\n  - not found item {item}")
                else:
                    original_distribution = self.processed_data[item].value_counts().to_dict()  # gets actual distribution of values as a dict
                    nas = self.processed_data[item].isna().sum()
                    original_distribution.update({'na': nas})

                    type_of_conversion = type[
                        type.iloc[:, 0] == item]
                    type_of_conversion = type_of_conversion.iloc[:, 1].iloc[0]

                    try:
                        self.processed_data.iloc[:, idx] = self.processed_data.iloc[:, idx].astype(type_of_conversion,
                                                                                                   errors='ignore')

                        # adding/removing (valid) categories
                        if type_of_conversion == 'category' and item in existing_categories:

                            actual_categories = self.processed_data.iloc[:, idx].values.categories.values
                            add_cat = [i for i in existing_categories[item] if i not in actual_categories]
                            remove_cat = [i for i in  actual_categories if i not in existing_categories[item]]

                            if len(add_cat) > 0:
                                self.processed_data.iloc[:, idx] = self.processed_data.iloc[:, idx].cat.add_categories(add_cat)  # add (unused) categories
                                # print(self.processed_data[item].values.categories.values)
                                logging.info(f"\n   - add additional categories to column '{item}': "
                                             f"{list(add_cat)}")
                            if len(remove_cat) > 0:
                                self.processed_data.iloc[:, idx] = self.processed_data.iloc[:, idx].cat.remove_categories(remove_cat)
                                # print(self.processed_data[item].values.categories.values)
                                logging.warning(f"\n   - removed categories from column '{item}' which were in data but not in existing "
                                                f"categories: {list(remove_cat)}")

                                # logging original__distribution vs. actual_distribution:
                                actual_distribution = self.processed_data[item].value_counts().to_dict()
                                nas = self.processed_data[item].isna().sum()
                                actual_distribution.update({'na': nas})
                                logging.warning(f"\n   - due removal of one or more categories in column '{item}' items have been removed too!"
                                                f"\n     original distribution: {original_distribution}"
                                                f"\n     actual distribution  : {actual_distribution}")

                        if self.processed_data.iloc[:, idx].dtypes != type_of_conversion:
                            logging.warning(
                                f"\n  - conversion of '{item}' (column {idx}) failed: type: {self.processed_data.iloc[:, idx].dtypes} but desired type: {type_of_conversion}")

                    except ValueError as er:
                        print(f"\n  - conversion failed: {er}")

            if len(missing_columns) != 0:
                logging.warning(f"\n  - No conversion: items not found: {missing_columns}")

    def replace_missing_values(self, missing_values=None, count_values=True):
        """ values defined as missing values in missing_values are replaced on df_data by 'nan'

        :param missing_values: Exp: mv = ["n/a", "na", "???", ".", "NR"]
        :param count_values: boolean, should a dictionary with counted values of replaced missing values be written in log file?
        """
        replaced_values = []
        logging.info(f"replace_missing_values(count_values={count_values}) called")

        if missing_values is not None:

            dict = {}  # stores updated missing values counts
            for item in missing_values:
                # counts number of replaced items
                if count_values:
                    nr = 0
                    for idx, column in enumerate(self.processed_data):
                        a = self.processed_data.iloc[:, idx].value_counts()  # counts all values in a column
                        if item in a.index:
                            nr = nr + a[item]
                    dict.update({item: nr})
                self.processed_data.replace(item, np.nan, inplace=True)
            if count_values:
                logging.info(f"\n  - replace_missing_values: {self._format_logging_output(dict, length=self._length_logging_messages)}")
                # logging.info(f"    replace_missing_values: {dict}")

    def get_dataframe(self):
        """ returns pandas DataFrame
        """
        return self.processed_data

    def get_store_path(self):
        return self._mydir

    def get_overall_summary(self):
        """shows stats for initals and processes file"""

        initial_data = self._initial_data.describe().T
        processed_data = self.processed_data.describe().T

        logging.info(f"\n  - initial data summary:")
        logging.info(self._format_logging_output(text=initial_data, length=1000))

        logging.info(f"\n  - processed data summary:")
        logging.info(self._format_logging_output(text=processed_data, length=1000))

    def get_missing_values_summary(self, threshold=0.0001, direction="above"):
        """shows a summary of number of all missing values per column with at least one missing value. If a threshold
        is defined only the columns above/below (parameter direction) this threshold will be shown.

        :param threshold: numeric between 0 and 1. Items above/below this threshold will be shown in the summary
        :param direction: one out of 'above' o 'below'
        :returns: pandas Data.Frame with statistics of missing values
        """

        logging.info(f"get_missing_values_summary(threshold={threshold}, direction={direction}) called")

        # Total missing values for each feature
        self.nas = self.processed_data.isnull().sum()
        self.nas = self.nas.sort_values(ascending=False)
        self.nas = self.nas.to_frame()
        self.nas.columns = ["na"]
        self.nas['total'] = self.processed_data.shape[0]
        self.nas['percentage'] = self.nas['na'] / self.nas['total']

        if direction == 'above':
            missing = self.nas[self.nas['percentage'] >= threshold]
        if direction == 'below':
            missing = self.nas[self.nas['percentage'] <= threshold]
            missing = missing.sort_values(by=['na'], ascending=True)

        logging.info(f"\n{self._format_logging_output(text=missing, length=1000)}")

        return missing

    # TODO: error in case of all values are valid (no nan!)
    def plot_missing_values_chart(self, number=10, filename=None):
        """plots a chart of missing values per column with a number of most or least missing values.

        :param number: positive 'number' shows x features with most missing values. If 'number' is negative the x features with least missing values will be displayed.
        :param filename: if a filename (string) is defined, the generated plot will be stored with this filename in the autogenerated folder (timestamp). The
        """
        logging.info(f"plot_missing_values_chart(number={number}, filename={filename}) called")

        if number > 0:
            number = min(self.processed_data.shape[1], number)
            msno.matrix(self.processed_data[self.nas[:number].index])
        else:
            number = min(self.processed_data.shape[0], abs(number))
            not_missing = [self.nas[(self.processed_data.shape[1] - number):self.processed_data.shape[1]].index]

            # reverse list
            n = []
            if len(not_missing) != 0:
                for i in not_missing[0][::-1]:
                    n.append(i)

                msno.matrix(self.processed_data[n])

        if filename is None:  # show/plot graphic
            plt.show()
        else:  # store graphic
            parts = filename.split(".")
            if len(parts) == 1:
                logging.error(f"\n  - no format for graphic defined. Plot could not be saved!")
            else:
                if parts[1] in ['png', 'pdf']:
                    plt.savefig(f'{self._mydir}/{filename}')
                    logging.info(f"\n  - plot successfully stored under {self._mydir}/{filename}")
                else:
                    logging.error(f"\n  - graphic format is unknown. Please choose from .pdf or .png")

    def recode(self, recoding_table):
        """ recodes values in df_data with values of recoding_table. Non existing columns defined in recoding_table are
        are listed as logging-warning.

        :param recoding_table: dictionary with column names as key and  dictionary with old and new values as value. Exp: recoding_table = {'gender': {'1': 'Male', '2': 'Female'},'vaccination_status': {'1': "Complete", '2': "Incomplete"}}
        """

        # TODO: recode {gender: {1: 'male', 2:'female', 3: 'male'}} should be possible

        logging.info(f"recode(recode_table) called")
        logging.info(f"\n  - recode_table [{len(recoding_table)}]: "
                     f"{self._format_logging_output(text=recoding_table, length=self._length_logging_messages)}")

        column_names = list(self.processed_data.columns)
        non_existing_column = []

        for column, new_values in recoding_table.items():

            if column not in column_names:
                non_existing_column.append(column)
            self.processed_data.replace({column: new_values}, inplace=True)

            # loggs if recoding table and values in raw_data are not equal:
            unique_values_table = self.processed_data[column].value_counts()
            if len(unique_values_table) == len(new_values):
                pass  # print("everything OK")
            else:
                if len(unique_values_table) > len(new_values):
                    logging.warning(f"\n  - column '{column}': Values appears in 'data' but not in the new defined key:value "
                                    f"\n  - data:\n {self._format_logging_output(unique_values_table.to_string())},"
                                    f"\n  - key:value: {self._format_logging_output(new_values)}")
                else:
                    logging.warning(f"\n  - column '{column}': Values appears in 'key:value' but not in data "
                                    f"\n  - data:\n {self._format_logging_output(unique_values_table.to_string())},"
                                    f"\n  - key:value: {self._format_logging_output(new_values)}")

        if len(non_existing_column) != 0:
            logging.warning(f"\n  - there have been columns which have't been recoded [{len(non_existing_column)}]:"
                            f"{self._format_logging_output(text=non_existing_column, length=self._length_logging_messages)}")

    def remain_columns(self, names=None):
        """ Only columns mentioned in parameter 'names' will be remained in DataFrame.

        :params names: which columns should stay in data? [list of strings]
        """

        logging.info(f"remain_columns(names) was called")

        if names is not None:

            column_names = list(self.processed_data.columns)
            logging.info(f"\n  - names = {self._format_logging_output(names, length=self._length_logging_messages)}")

            if len(names) == 0:
                logging.warning(f"\n  - 'names' is empty! DataFrame would be empty as well after executing method! Method was NOT executed!")
            else:
                intersection = [value for value in names if value in column_names]
                diff = [value for value in names if value not in column_names]

                if len(intersection) == 0:
                    logging.warning(
                        f"\n  - no intersection of header and 'names'! DataFrame would be empty after executing method! Method was NOT executed!")
                else:
                    self.processed_data = self.processed_data[intersection]
                    logging.info(f"\n  - columns in data after execution [{len(intersection)}]: "
                                 f"{self._format_logging_output(intersection, length=self._length_logging_messages)}")
                    if len(diff) != 0:
                        logging.warning(f"\n  - these items from 'names' have not been in data [{len(diff)}]: "
                                        f"{self._format_logging_output(diff, length=self._length_logging_messages)}")
        else:
            logging.info(f"\n  - names = None")
            logging.warning(f"\n  - method was called with no parameter 'names' (None). Method was not executed!")

    def delete_columns(self, names):
        """ deletes columns specified in list 'names'.

        :param names: which columns should be deleted from df_data? [list of strings]
        """
        logging.info(f"delete_columns(names) called")

        column_names = list(self.processed_data.columns)
        intersection_list = [value for value in names if value in column_names]
        diff_list = [value for value in names if value not in column_names]

        logging.info(f"\n  - names [{len(names)}]: {self._format_logging_output(names, length=self._length_logging_messages)}")
        logging.info(f"\n  - shape before: {self.processed_data.shape}")

        if len(diff_list) > 0:
            logging.warning(f"\n  - there have been columns [{len(names) - len(intersection_list)}] "
                            f"which do not exist in df_data: {self._format_logging_output(diff_list, length=self._length_logging_messages)}")

        self.processed_data.drop(intersection_list, axis=1, inplace=True)
        logging.info(f"\n  - shape after: {self.processed_data.shape}")

    def delete_columns_missing_values(self, threshold=1):
        """ deletes columns if fraction of missing values is above defined threshold.

        :param threshold: numeric value between 0 and 1. What should be the fraction of missing data in a column to drop this?
        """

        logging.info(f"delete_columns_missing_values(threshold={threshold}) called")

        columns = self.processed_data.columns
        percent_missing = self.processed_data.isnull().sum() / len(self.processed_data)
        missing_value_df = pd.DataFrame({'column_name': columns,
                                         'percent_missing': percent_missing.values})

        missing = missing_value_df[missing_value_df['percent_missing']>threshold].sort_values(by=['percent_missing'], ascending=False)
        missing_drop = list(missing['column_name'])

        # missing_drop = list(missing_value_df[missing_value_df.percent_missing > threshold].column_name)

        if len(missing_drop) >= 1:
            logging.warning(f"\n  - columns droped [{len(missing_drop)}]: {self._format_logging_output(missing, length=self._length_logging_messages)}")
        else:
            logging.info(f"\n  - no columns have been droped!")

        self.processed_data = self.processed_data.drop(missing_drop, axis=1)

    def save_data_frame(self, filename=None):
        """ saves modified DataFrame as csv- or pickle-file with the given filename in autocreated folder (timestamp).
            The pickle-file (pkg) keeps column type and more and can be read-in again for further analysis.

        :param filename: string (exp. 'test.csv' or 'test.pkl')
        """
        if filename is None:
            filename = "final_data.csv"
        logging.info(f"save_data_frame(filename={filename}) called")

        parts = filename.split(".")
        if parts[1] == 'csv':
            self.processed_data.to_csv(f"{self._mydir}/{filename}", index=False)
            logging.info(f"\n  - csv-file successfully saved")
        else:
            if parts[1] == 'pkl':
                self.processed_data.to_pickle(f"{self._mydir}/{filename}")
                logging.info(f"\n  - pkl-file successfully saved")
            else:
                logging.warning(f"\n  - not recognized file ending. Nothing saved!")

    def _move_text(self, text):
        space = ' '  # specifying delimiter
        text = f"\n{(sum(self._format_logging) + 31) * space}{text}"
        return text

    @staticmethod
    def _convert_to_string(data):
        if data is not None:
            return data.astype(str)  # reads in all columns as type string/object

    @staticmethod
    def arrange_columns_alphabetically(df, case_sensitivity=False, order="AZ"):
        """ arranges columns alphabetically by name. This can be made case-sensitive or non case-sensitive.

        :param df: pandas DataFrame
        :param case_sensitivity: boolean, should alphabetically order be case-sensitive or not?
        :param order: string, one out of 'AZ' or 'ZA'
        :return: pandas DataFrame with new arranged columns
        """

        if order == 'AZ':
            reverse = False
        else:
            if order == 'ZA':
                reverse = True
            else:
                print("order is not valid and was set to 'AZ'")
                reverse = False

        col_names = list(df.columns.values)  # save original order in columns

        if case_sensitivity is True:
            # case sensitive
            df_rearranged = df.reindex(sorted(df.columns, reverse=reverse), axis=1)
        else:
            # non case sensitive
            lower_col_names = [x.lower() for x in col_names]
            data_tuples = list(zip(lower_col_names, col_names))

            df_colnames = pd.DataFrame(data_tuples)

            sorted_df_colnames = (df_colnames.sort_values(by=0, ascending=not reverse))
            dfToList = sorted_df_colnames[1].tolist()
            df_rearranged = df[dfToList]

        return df_rearranged

    @staticmethod
    def create_recoding_dict(df_coding):
        """ creates a dictionary with {column name: {old value: new vale}} to feed method 'recode' with. The method
        involves only variables of type 'category' (type)
        exp.: recoding_table = {
              'gender': {'1': 'Male', '2': 'Female'},
              'vaccination_status': {'1': "Complete", '2': "Incomplete"}
            }

        :param df_coding: pandas DataFrame with columns 'header', 'type' and 'value_range' (positional not name sensitive)
        :return: dictionary to feed method 'recode'
        """
        recoding_table = {}  # stores updated missing values counts
        data = df_coding[df_coding.iloc[:, 1] == 'category']
        notnull = pd.notnull(data.value_range)  # takes only items with valid entry in 'value_range'
        data = data[notnull]

        for row in data.iterrows():
            result = [x.strip() for x in row[1]['value_range'].split(',')]
            dict_inner = {}
            for item in result:
                r = [x.strip() for x in item.split('=')]
                dict_inner.update({r[0]: r[1]})
            recoding_table.update({row[1]['header']: dict_inner})

        return recoding_table

    def remove_text_from_columns(self, colnames):
        """ Deletes text from as numeric defined columns

        :param colnames: list of strings with name of columns which should be treated
        """

        logging.info(f"remove_text_from_columns({list(colnames)}) was called")
        for column in colnames:
            data = self.processed_data[column].fillna('999')
            for idx, item in enumerate(data):
                try:
                    float(item)
                    if item == '999':
                        data.iloc[idx] = np.nan
                except ValueError:
                    logging.warning(f"\n  - value '{item}' in column '{column}' [{idx}] was set to np.nan!")
                    data.iloc[idx] = np.nan

            self.processed_data[column] = data.values  # update original data

    def _format_logging_output(self, text, length=100):

        space = ' '  # specifying delimiter
        separator = ' '  # what possible chars should be replaced be '\n'?

        text_to_modify = text

        if isinstance(text_to_modify, list) or isinstance(text_to_modify, dict):
            text_to_modify = str(text_to_modify)
            separator = ','

            if "\n" not in text:  # add \n at specific positions in text

                # position of all separators in text
                position_separator = ([pos for pos, char in enumerate(text_to_modify) if char == separator])

                if len(position_separator) == 0:
                    return text

                desired_break_position = list(range(0, position_separator[-1], length))
                real_break_positions = []

                if len(desired_break_position) == 1:
                    return text  # no modification required due text is shorter as length
                else:
                    for item in desired_break_position:
                        break_position = [i for i in position_separator if i < item]
                        if len(break_position) > 0:
                            real_break_positions.append(break_position[-1])  # append last possible break_position

                    real_break_positions.insert(0, 0)  # insert 0 at first place of list

                    mutable_text = list(text_to_modify)
                    for item in real_break_positions:
                        mutable_text[item] = ',\n'
                        changed_text = ''.join(mutable_text)

                    text = changed_text

        if isinstance(text_to_modify, pd.DataFrame):
            text = text_to_modify.to_string()

        # move text to right side (insert spaces)
        parts = text.splitlines()  # splits at \n
        for idx, item in enumerate(parts):
            if idx == 0:
                parts[idx] = f"   |{parts[idx]}"
            else:
                parts[idx] = f"    |{parts[idx]}"

            # if idx > 0:
            #     parts[idx] = f"   |{parts[idx]}"

        final_text = '\n'.join(parts)  # join all parts of the list with \n as delimiter

        return final_text

    ###################################################################################################################
    # not needed anymore!
    def _format_logging_output2(self, text, length=100):
        """ formats strings, lists and pandas DataFrame for better readability for logging output_2020-04-29_19-41-55. If text already contains '\n'
        it will only be shifted for logging output_2020-04-29_19-41-55 by defined positions in self._format_logging (exp. output_2020-04-29_19-41-55 of
        statistics). If text is a list, the items of this list will be separated first by ','. Then '\n' are
        automatically insert so that each line has a maximal length which is defined in parameter 'length'.

        :param text: string, list or pandas DataFrame which should be converted
        :param length: if text is a list, linebreaks will be automatically insert so that each line in output_2020-04-29_19-41-55 has a maximal length of parameter 'length'
        :return: a formatted string with insert line changes
        """

        space = ' '  # specifying delimiter
        separator = ' '  # what possible chars should be replaced be '\n'?

        text_to_modify = text

        if isinstance(text_to_modify, list) or isinstance(text_to_modify, dict):
            text_to_modify = str(text_to_modify)
            separator = ','

            if "\n" not in text:  # add \n at specific positions in text

                # position of all separators in text
                position_separator = ([pos for pos, char in enumerate(text_to_modify) if char == separator])

                if len(position_separator) == 0:
                    return text

                desired_break_position = list(range(0, position_separator[-1], length))
                real_break_positions = []

                if len(desired_break_position) == 1:
                    return text  # no modification required due text is shorter as length
                else:
                    for item in desired_break_position:
                        break_position = [i for i in position_separator if i < item]
                        if len(break_position) > 0:
                            real_break_positions.append(break_position[-1])  # append last possible break_position

                    mutable_text = list(text_to_modify)
                    for item in real_break_positions:
                        mutable_text[item] = ',\n'
                        changed_text = ''.join(mutable_text)

                    text = changed_text

        if isinstance(text_to_modify, pd.DataFrame):
            text = text_to_modify.to_string()

        # move text to right side (insert spaces)
        parts = text.splitlines()  # splits at \n
        for idx, item in enumerate(parts):
            if idx is 0:  # first element should not be changed
                parts[0] = f"{4 * space}{parts[idx]}"  # 3 spaces for first line
            else:
                parts[
                    idx] = f"{(sum(self._format_logging) + 31) * space}{parts[idx]}"  # 31 -> length of logging date (without milliseconds)
        final_text = '\n'.join(parts)  # join all parts of the list with \n as delimiter

        return final_text