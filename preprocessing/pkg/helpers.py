# helper functions for SOSALL_preprocessing

import plotly.graph_objs as go
import pandas as pd
import itertools
import numpy as np
import re


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
    return(df_multi_items)


# remark: nan will be converted to 0 in this function!
# remark: if a value is not in coding-table then it will be switched to nan or 0 resp. as well

def dummy_coding_multiple_items_entries(data, colnames, coding=None, split=';', dummy_coding=[0,1]):

    store = []

    colnames = [item for item in colnames if item in data.columns]  # take these columns existing in data

    # for each column unique values will be taken to make matrix with original column-name supplemented by _(value)
    for idx, item in enumerate(colnames):
        df = data[item].to_frame()
        # df=df.replace("", np.NaN)
        df=df.replace(np.NaN, "")

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
            postfix = [coding[idx].get(key) for key in uniques if key in coding[idx].keys()]  # get only these entries which are in coding (key) too
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


def table_unique_values(df):

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
    df.loc[df['foodreaction_any'] != 'no',  'foodreaction_any'] = 'yes'

    data['foodreaction_any'] = df['foodreaction_any']
    data.drop(cols_foodreaction, axis=1, inplace=True)
    return data


# 1=none, 2=asthma, 3=hayfever, 4=eczema, 5=food allergy

# new coding:
# if 4=eczema in familyhistory_mother -> NEW: familyhistory_mother_eczema is 'yes'
# if 2,3,5 (NOT 'none' or 'eczema') in familyhistory_mother -> NEW: familyhistory_mother_other_allergic_disease is 'yes'
# same for familyhistory_father and familyhistory_siblings
# in case of siblings: loop for all 4 siblings but reduced to one column -> if at least one of these 4 siblings has
# eczema (3) then NEW familyhistory_sibling_eczema' will be 'yes'
# (same for 'familyhistory_sibling_other_allergic_disease'?

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

        if x['sp_neg_control']==0 and x['sp_positive_control'] !=0:
            items = x.drop(labels=['sp_neg_control','sp_positive_control'])
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