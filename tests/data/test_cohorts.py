import numpy as np
import pandas as pd

import seismometer.data.cohorts as undertest


def input_df():
    return pd.DataFrame(
        {
            "TARGET": [1, 0, 0, 1, 0, 1],
            "col1": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "tri": [0.0, 0, 1, 1, 2, 2],
        }
    )


def expected_df(cohorts):
    data_rows = np.vstack(
        (
            # TP,FP,TN,FN,   Acc,Sens,Spec,PPV,NPV,    Flag,      LR+,  cohort,ct,tgtct
            [[0, 0, 1, 1, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, np.nan, "<1.0", 2, 1]] * 70,
            [[0, 1, 0, 1, 0, 0, 0, 0, 0, 0.5, 0, "<1.0", 2, 1]] * 10,
            [[1, 1, 0, 0, 0.5, 1, 0, 0.5, 1, 1, 1, "<1.0", 2, 1]] * 21,
            # TP,FP,TN,FN,   Acc,Sens,Spec, PPV, NPV,    Flag,      LR+,  cohort,ct,tgtct
            [[0, 0, 2, 2, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, np.nan, ">=1.0", 4, 2]] * 30,
            [[1, 0, 2, 1, 0.75, 0.5, 1, 1, 2 / 3, 0.25, np.inf, ">=1.0", 4, 2]] * 10,
            [[1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, ">=1.0", 4, 2]] * 10,
            [[2, 1, 1, 0, 0.75, 1, 0.5, 2 / 3, 1, 0.75, 2, ">=1.0", 4, 2]] * 10,
            [[2, 2, 0, 0, 0.5, 1, 0, 0.5, 1, 1, 1, ">=1.0", 4, 2]] * 41,
            # TP,FP,TN,FN,   Acc,Sens,Spec,PPV,NPV,    Flag,      LR+,  cohort,ct,tgtct
            [[0, 0, 1, 1, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, np.nan, "1.0-2.0", 2, 1]] * 50,
            [[1, 0, 1, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, np.inf, "1.0-2.0", 2, 1]] * 10,
            [[1, 1, 0, 0, 0.5, 1, 0, 0.5, 1, 1, 1, "1.0-2.0", 2, 1]] * 41,
            # TP,FP,TN,FN,   Acc,Sens,Spec,PPV,NPV,    Flag,      LR+,  cohort,ct,tgtct
            [[0, 0, 1, 1, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, np.nan, ">=2.0", 2, 1]] * 30,
            [[1, 0, 1, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, np.inf, ">=2.0", 2, 1]] * 10,
            [[1, 1, 0, 0, 0.5, 1, 0, 0.5, 1, 1, 1, ">=2.0", 2, 1]] * 61,
        )
    )

    df = pd.DataFrame(
        data_rows,
        columns=[
            "TP",
            "FP",
            "TN",
            "FN",
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "PPV",
            "NPV",
            "Flagged",
            "LR+",
            "cohort",
            "cohort-count",
            "cohort-targetcount",
        ],
    )
    df["Threshold"] = np.tile(np.arange(100, -1, -1), 4)

    df[["TP", "FP", "TN", "FN", "cohort-count", "cohort-targetcount"]] = df[
        ["TP", "FP", "TN", "FN", "cohort-count", "cohort-targetcount"]
    ].astype(int)
    df[["Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Flagged", "LR+"]] = df[
        ["Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Flagged", "LR+"]
    ].astype(float)

    # reduce before creating category
    reduced_df = df.loc[df.cohort.isin(cohorts)].reset_index(drop=True)
    reduced_df["cohort"] = pd.Categorical(reduced_df["cohort"], categories=cohorts)
    return reduced_df


class Test_Performance_Data:
    def test_data_defaults(self):
        df = input_df()
        actual = undertest.get_cohort_performance_data(df, "tri", proba="col1", censor_threshold=0)

        expected = expected_df(["<1.0", ">=1.0"])

        pd.testing.assert_frame_equal(actual, expected, check_column_type=False, check_like=True, check_dtype=False)

    def test_data_splits(self):
        df = input_df()
        actual = undertest.get_cohort_performance_data(df, "tri", proba="col1", splits=[1.0, 2.0], censor_threshold=0)

        expected = expected_df(["<1.0", "1.0-2.0", ">=2.0"])

        pd.testing.assert_frame_equal(actual, expected, check_column_type=False, check_like=True, check_dtype=False)
