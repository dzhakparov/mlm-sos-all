from config import target, subset
from statTests.statSuite import StatSuite
import pandas as pd


def run(output_folder):

    data = pd.read_pickle(f"{output_folder}/final_data.pkl")

    if subset != "both":
            data = data[data['country'] == subset]

    if target == 'diagnosis':
        data.drop(['diagnosis_location'], axis=1, inplace=True)
    else:
        data.drop(['diagnosis'], axis=1, inplace=True)

    obj = StatSuite(data=data, target=target, path=output_folder)

    obj.run(store_csv=True)
    obj.plot(interactive=True)


if __name__ == '__main__':
    run(output_folder="../data/output/2022-03-14_15:16_diagnosis_sp_any_True")
