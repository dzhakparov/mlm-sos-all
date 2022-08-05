from src.old.config import output_folder, sp_any, target, subset
from shutil import copyfile
import pandas as pd
import plotly.graph_objects as go
import os
from pathlib import Path
import itertools
import numpy as np
import re
from os import listdir
from os.path import isfile, join
from datetime import datetime
import glob


def store_data(train_data, test_data):
    data = [(train_data, "train_data"), (test_data, "test_data")]
    for item, name in data:
        if item is not None:
            item.to_csv(f"{output_folder}/{name}.csv", index=True, header=True)
            item.to_pickle(f"{output_folder}/{name}.pkl")
    print("stored data_old successfully!")


def get_working_dir(dt_string=None):
    if dt_string:
        folder = f"{output_folder}/{dt_string}_{target}_{subset}_sp_any_{sp_any}"
    else:  # newest directory as working dir
        dirs = listdir(output_folder)
        if dirs:
            folder = max(glob.glob(os.path.join(output_folder, '*')), key=os.path.getmtime)
        else:
            raise Exception(f"could not find a matching working directory!")
    return folder


def build_working_directory(dt_string=None):
    dirs = listdir(output_folder)
    if dt_string is None and not dirs:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H:%M")
        folder = f"{output_folder}/{dt_string}_{target}_{subset}_sp_any_{sp_any}"
    elif dt_string is not None:
        folder = f"{output_folder}/{dt_string}_{target}_{subset}_sp_any_{sp_any}"
    elif dirs:  # newest directory as working dir
        folder = max(glob.glob(os.path.join(output_folder, '*')), key=os.path.getmtime)
    else:
        raise Exception(f"could not build a valid working directory")
    return folder


def set_up_infrastructure(dt_string=None):
    folder = build_working_directory(dt_string=dt_string)
    Path(folder).mkdir(parents=True, exist_ok=True)
    return folder


def copy_config(output_folder):
    copyfile(os.path.join(os.getcwd(), "config.py"), os.path.join(output_folder, "config.py"))


def set_wd():
    # sets working directory always to main.py location. This is useful due to find raw input data_old if main.py is
    # executed but also when a single file is executed
    act_dir = os.getcwd()
    onlyfiles = [f for f in listdir(act_dir) if isfile(join(act_dir, f))]
    if 'main.py' not in onlyfiles:
        os.chdir("../../")


def plot_missing_values_per_group(data, target):
    df = data.groupby(target).agg(['size', 'count'])
    df = df.swaplevel(0, 1, axis=1)
    df = df.unstack(level=1)
    df = df.swaplevel(0, 2)
    df = df.unstack(level=2)
    df = df.swaplevel(0, 1)
    df.sort_index(axis=0, level=0, inplace=True)
    df['na_percentage'] = 1 - (df['count'] / df['size'])

    fig = go.Figure()
    for name, df in df.groupby(target):
        fig.add_trace(go.Bar(
            x=df.index.levels[0],
            y=df['na_percentage'],
            name=name,
        ))
        fig.update_layout(
            title={
                'text': 'missing values per group',
                'font': {'size': 25}
            },
            template='plotly_white'
        )
    return fig


def _build_summary_data(data, target):
    df = data.groupby(target).agg(['mean', 'median', 'min', 'max', 'std', 'size', 'count'])
    df = df.swaplevel(0, 1, axis=1)
    df = df.unstack(level=1)
    df = df.swaplevel(0, 2)
    df = df.unstack(level=2)
    df = df.swaplevel(0, 1)
    df.sort_index(axis=0, level=0, inplace=True)
    df['na_percentage'] = 1 - (df['count'] / df['size'])
    return df


def plot_heatmap(data):
    data = pd.isna(data)  # all values are 0 except missing values are 1
    no_missing_values = sum((data == 1).sum(axis=1))
    total_values = data.shape[0] * data.shape[1]

    store = []
    for idx, col in enumerate(data):
        temp = data.iloc[:, idx]
        temp = temp.to_frame(name='value')
        temp = temp.reset_index()
        temp['id'] = temp['index']
        temp['index'] = temp.index
        temp['colname'] = col
        store.append(temp)

    data = pd.concat(store)

    # convert type of column 'value' from bool to int (for heatmap)
    data['value'] = data['value'].astype('int64')

    data = [go.Heatmap(x=data['colname'],
                       y=data['index'],
                       z=data['value'],
                       text=data['id'],
                       hovertemplate='id: %{text}<br>x: %{x}<br>missing: %{z}<br><extra></extra>',
                       zmin=0,
                       zmax=1,
                       showscale=False  # hides colorscale
                       )]
    layout = go.Layout(
        title=f"missing values: {no_missing_values} ( = yellow colored) from  {total_values} "
              f"({round(no_missing_values / total_values * 100, 3)}%)",
        height=800
    )
    fig = go.Figure(data=data, layout=layout)
    return fig


def multiple_items(df):
    colnames = df.columns

    multi_items = []
    for idx, item in enumerate(colnames):
        if not df[item].isnull().all():
            bool = df[item].str.contains(';')
            bool = bool.fillna(False)
            temp = df[bool][item].to_frame()
            if len(temp) > 0:
                temp = {
                    'column': list(temp.to_dict().keys())[0],
                    'no entries not nan': len(df) - df[item].isnull().sum(),
                    'no multiple entries': len(temp),
                    'unique values': len(df[item].unique()),
                    'items': list(temp.to_dict().values())[0]
                }
                multi_items.append(temp)

    df_multi_items = pd.DataFrame(multi_items)
    return (df_multi_items)


def dummy_coding_multiple_items_entries(data, colnames, coding=None, split=';', dummy_coding=[0, 1]):
    store = []

    colnames = [item for item in colnames if item in data.columns]  # take these columns existing in data_old

    # for each column unique values will be taken to make matrix with original column-name supplemented by _(value)
    for idx, item in enumerate(colnames):
        df = data[item].to_frame()
        # df=df.replace("", np.NaN)
        df = df.replace(np.NaN, "")

        if not df[item].isnull().all():
            # single cells without ;
            bool = df[item].str.contains(split).fillna(False)
            temp = df[np.where(bool, False, True)]
            uniques_single = temp[item].unique().tolist()
            # uniques_single = [str(string) for string in uniques_single if string != "" and str(string) != 'nan']  # remove '' from list
            uniques_single = [str(string) for string in uniques_single if string != ""]  # remove '' from list

            # multiple cells with ;
            temp = df[bool][item].to_frame()

            uniques_list = []
            for index, row in temp.iterrows():
                uniques_list.append(row[item].split(split))

            uniques_multiple = list(set(itertools.chain.from_iterable(uniques_list)))  # uniques = ['2', '3', '1', '4']
            joined_list = uniques_single + uniques_multiple
            uniques = list(set(joined_list))
        else:
            uniques = df[item].uniques().tolist()

        # get values from coding to according keys
        if coding is None:
            postfix = uniques
        else:
            postfix = [coding[idx].get(key) for key in uniques if
                       key in coding[idx].keys()]  # get only these entries which are in coding (key) too
            uniques = [i for i in uniques if i in coding[idx].keys()]

        # add uniques to colnames and build matrix
        new_colnames = [item + '_' + i for i in postfix]

        dff = pd.DataFrame(index=data.index, columns=range(len(new_colnames)))
        dff.columns = new_colnames

        # populate new DataFrame with information from item
        for idx, row in df.iterrows():
            a = row[item].split(';')
            if len(a) == 1:
                if a[0] != '' and a[0] in uniques:
                    index = uniques.index(a[0])
                    dff.iloc[idx, index] = dummy_coding[1]
                else:
                    if a[0] not in uniques:
                        pass  # TODO: register that value is converted to nan due its not in coding list
            else:
                for i in a:
                    index = uniques.index(i)
                    dff.iloc[idx, index] = dummy_coding[1]

        # replace nan by 0
        dff = dff.fillna(dummy_coding[0])

        # store dff in list
        store.append(dff)

    # remove old columns and replace them by new build matrix
    data.drop(colnames, axis=1, inplace=True)

    for item in store:
        data = pd.concat([data, item], axis=1)

    return data


def table_unique_values(df, width=500):
    data = df.copy()

    all_columns = list(data)  # Creates list of all column headers
    v_types = data.dtypes
    v_na = data.isna().sum()
    data[all_columns] = data[all_columns].astype(str)
    v_unique = data.nunique()
    v_total = data.shape[0]
    v_relative = v_na / v_total

    counts = []
    for col in data:
        count_all = data[col].value_counts()
        if 'nan' in count_all:
            count_all.drop(labels=['nan'], inplace=True)
            v_unique[col] = v_unique[col] - 1
        if len(count_all) >= 7:
            a = str(count_all[0:3].to_dict())  # take first 3 and last 3 items
            b = str(count_all.tail(3).to_dict())
            c = a + ' ... ' + b
            characters_to_remove = "{}"
            pattern = "[" + characters_to_remove + "]"
            c = re.sub(pattern, "", c)
            counts.append(c)
        else:
            a = str(count_all.to_dict())
            characters_to_remove = "{}"
            pattern = "[" + characters_to_remove + "]"
            a = re.sub(pattern, "", a)
            counts.append(a)

    values = pd.Series(counts, index=v_unique.index.tolist())
    dff = pd.DataFrame(dict(unique=v_unique,
                            # types=v_types,
                            values=values,
                            NA=v_na,
                            relative_NA=v_relative,
                            total=v_total
                            )).reset_index()

    dff.style.set_properties(subset=['values'], **{'width': f'{width}px'})
    dff = dff.sort_values(by=['relative_NA'], ascending=False)
    return dff


def recoding_foodreaction(data):
    cols_foodreaction_ever = [x for x in data.columns if 'foodreaction' in x and 'ever' in x]  # (7)
    cols_foodreaction = [x for x in data.columns if "foodreaction" in x]  # all columns with 'foodreaction' (35)
    df = data[cols_foodreaction_ever].fillna(0)
    df = df.replace('no', 0)
    df = df.apply(pd.to_numeric)
    df['foodreaction_any'] = df.sum(axis=1)
    df.loc[df['foodreaction_any'] == 0, 'foodreaction_any'] = 'no'
    df.loc[df['foodreaction_any'] != 'no', 'foodreaction_any'] = 'yes'

    data['foodreaction_any'] = df['foodreaction_any']
    data.drop(cols_foodreaction, axis=1, inplace=True)
    return data


def recoding_familyhistory(data):
    columns = ['familyhistory_mother', 'familyhistory_father', 'familyhistory_sibling1', 'familyhistory_sibling2',
               'familyhistory_sibling3', 'familyhistory_sibling4']

    data = data.join(pd.DataFrame(
        {
            'familyhistory_mother_eczema': 'no',
            'familyhistory_mother_other_allergic_disease': 'no',
            'familyhistory_father_eczema': 'no',
            'familyhistory_father_other_allergic_disease': 'no',
            'familyhistory_sibling_eczema': 'no',
            'familyhistory_sibling_other_allergic_disease': 'no',
        }, index=data.index
    ))

    for column in columns:

        if 'familyhistory_sibling' in column:
            column_group = 'familyhistory_sibling'
        else:
            column_group = column

        for idx, row in data.iterrows():
            if isinstance(row[column], str):
                entries = row[column].split(';')
                for entry in entries:
                    if entry == '4':
                        data.loc[idx, [column_group + '_eczema']] = 'yes'
                    else:
                        if entry != '1':
                            data.loc[idx, [column_group + '_other_allergic_disease']] = 'yes'

    # remove 'original' columns:
    data.drop(columns, axis=1, inplace=True)

    return data


def recoding_sp(data):
    df = data.copy()
    cols_sp = [x for x in df.columns if 'sp_' in x]  # (12)

    def set_sp_value(x):
        x = pd.to_numeric(x)

        if x.isnull().values.all():
            return np.nan

        if x['sp_neg_control'] == 0 and x['sp_positive_control'] != 0:
            items = x.drop(labels=['sp_neg_control', 'sp_positive_control'])
            if sum(items) > 0:
                return 'yes'
            else:
                return 'no'
        else:
            return 'no'

    df['sp_any'] = df[cols_sp].apply(set_sp_value, axis=1)

    df.drop(cols_sp, axis=1, inplace=True)
    # df = df[[x for x in df.columns if 'sp_' in x]]

    return df, cols_sp


def recoding_immun(data):
    df = data.copy()

    cols_immun = [x for x in df.columns if 'immun_' in x]

    def set_immun_value(x):
        x = pd.to_numeric(x)
        if x.isnull().values.all():
            return np.nan
        else:
            return str(np.nansum(x))

    df['immun_number'] = df[cols_immun].apply(set_immun_value, axis=1)

    df.drop(cols_immun, axis=1, inplace=True)
    # df = df[[x for x in df.columns if 'immun_' in x]]

    return df, cols_immun


# TODO: maybe include in package PredictionPipeline as staticmethod...
def build_recode_object(table):
    data = table[table.iloc[:, 1] == 'category']  # type (column 2) has to be 'category'
    notnull = pd.notnull(data.value_range)  # takes only items with valid entry in 'value_range'
    data = data[notnull]
    results = []
    for row in data.iterrows():
        result = [x.strip() for x in row[1]['value_range'].split(',')]
        dict_inner = {}
        for item in result:
            r = [x.strip() for x in item.split('=')]
            dict_inner.update({r[0]: r[1]})
        results.append([dict_inner, [row[1]['header']]])
    return results
