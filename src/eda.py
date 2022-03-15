# EDA = Exploratory Data Analysis

import config
import pandas as pd
from pathlib import Path
from src.helpers import plot_missing_values_per_group

from misc.df_plotter.mv_plotter import MissingValuePlotter
from misc.df_plotter.column_plotter import ColumnPlotter
from misc.df_plotter.type_plotter import TypePlotter
from misc.df_plotter.box_plotter import BoxPlotter


def run(output_folder):

    try:
        Path(f"{output_folder}/eda/").mkdir(parents=True, exist_ok=True)
        output_path = f"{output_folder}/eda/"
    except FileNotFoundError:
        raise Exception(f"could not create folder {output_folder}/eda")

    data = pd.read_pickle(f"{output_folder}/final_data.pkl")

    if config.subset != "both":
        data = data[data['country'] == config.subset]

    seqs_list = list(range(0, data.shape[1], 16)) + [data.shape[1]]
    seqs = []
    for idx, item in enumerate(seqs_list):
        if idx > 0:
            seqs.append((seqs_list[idx - 1], item))

    for idx, seq in enumerate(seqs):
        fig = ColumnPlotter(df=data.iloc[:, seq[0]:seq[1]], nrows=4,
                            ncols=4).get_plot()  # TODO: add case with more than 49 subplots in ColumnPlotter
        fig.write_html(f"{output_path}/ColumnPlotter_{idx}.html")

    fig = MissingValuePlotter(df=data).get_plot()
    fig.write_html(f"{output_path}/MissingValuePlotter.html")

    fig = TypePlotter(df=data, sort='missing').get_plot()
    fig.write_html(f"{output_path}/TypePlotter.html")

    bp = BoxPlotter(df=data, features=None, split=config.target)
    bp.update_layout(n_cols=3, n_rows=3, style_grid={'vertical_spacing': 0.1, 'horizontal_spacing': 0.1},  # TODO: add to config or BoxPlot respectivly
                     colors={'AD': 'green', 'HC': 'red'}, style_figure={'boxpoints': 'all'})
    figs = bp.get_plot()
    bp.store(path=output_path, name='BoxPlotter')

    fig = plot_missing_values_per_group(data, config.target)
    fig.write_html(f"{output_path}/mv_per_group.html")


if __name__ == '__main__':
    run(output_folder="../data/output/2022-03-14_15:16_diagnosis_sp_any_True")
