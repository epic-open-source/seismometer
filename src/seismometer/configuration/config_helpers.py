from pathlib import Path

import pandas as pd

from seismometer.core.io import write_yaml

from .model import DictionaryItem, EventDictionary, PredictionDictionary


def generate_dictionary_from_parquet(
    inpath: Path | str, outpath: Path | str, section: str = "predictions", *, column: str = "Type"
) -> None:
    """
    Generate a data dictionary YAML file from a parquet file.

    Parameters
    ----------
    inpath : Path | str
        The path to the input Parquet file.
    outpath : Path | str
        The path to the output YAML file.
    section : str, optional
        The section name to be used in the YAML file, by default "predictions".
    column : str, optional
        The column name of data relevant to defining the dictionary; currently used
        for events section, by default "Type".
    """
    df = pd.read_parquet(inpath)
    if df.empty:
        raise ValueError("No data loaded; check the input file")

    datadict = None
    match section:
        case "predictions":
            datadict = _generate_prediction_dictionary(df)
        case "events":
            datadict = _generate_event_dictionary(df, column)
        case _:
            raise ValueError(f"Section {section} not recognized; only supports predictions and events")

    write_yaml(datadict.model_dump(), outpath)


def _generate_prediction_dictionary(df: pd.DataFrame) -> PredictionDictionary:
    """Generates dictionary based on columns in the DataFrame"""
    items = [
        DictionaryItem(name=c, dtype=str(df[c].dtype), definition=f"Placeholder description for {c}")
        for c in df.columns
    ]
    return PredictionDictionary(predictions=items)


def _generate_event_dictionary(df, column="Type") -> EventDictionary:
    """Generates entry for each unique value in the specified column"""
    items = (
        [
            DictionaryItem(name=c, dtype="string", definition=f"Placeholder description for {c}")
            for c in df[column].unique()
        ]
        if column in df
        else []
    )
    return EventDictionary(events=items)
