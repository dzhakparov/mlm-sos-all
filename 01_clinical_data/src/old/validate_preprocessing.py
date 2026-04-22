# This script validates the preprocessed file with
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check, Index


def run_validation(df):
    schema = DataFrameSchema(
        {
            "PID": Column(str),
            "diagnosis": Column()
            # "column2": Column(float, Check(lambda s: s < -1.2)),
            # # you can provide a list of validators
            # "column3": Column(str, [
            #     Check(lambda s: s.str.startswith("value")),
            #     Check(lambda s: s.str.split("_", expand=True).shape[1] == 2)
            # ]),
        },
        # index=Index(int),
        # strict=True,
        # strict='filter',
        # coerce=False,
    )
    return schema.validate(df)


if __name__ == '__main__':
    # df = pd.read_pickle("../data/output/data_processed.pkl")
    # desc = pd.read_csv("../data/input/description_data_ms9_kb2.csv", index_col=False)
    # desc = desc.loc[:,["working_header", "correct_type", "use", "value_range_modified"]]
    #
    # run_validation(df)

    from pandera import Column, DataFrameSchema

    df = pd.DataFrame({"column1": ["drop", "me"], "column2": ["keep", "me"]})
    schema = DataFrameSchema({"column2": Column(str)})

    validated_df = schema.validate(df)
    print(validated_df)
