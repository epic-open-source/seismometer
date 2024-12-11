from typing import Dict, List, Union

from seismometer.data.binary_performance import GENERATED_COLUMNS


class AnalyticsTableConfig:
    """
    Configuration class for table settings in the AnalyticsTable class.
    """

    def __init__(
        self,
        *,
        decimals: int = 3,
        columns_show_percentages: Union[str, List[str]] = "Prevalence",
        columns_show_bar: Dict[str, str] = None,
        percentages_decimals: int = 0,
        alternating_row_colors: bool = True,
        data_bar_stroke_width: int = 4,
    ):
        self.decimals = decimals
        self.columns_show_percentages = columns_show_percentages
        self.columns_show_bar = columns_show_bar or {}

        self.percentages_decimals = percentages_decimals
        self.alternating_row_colors = alternating_row_colors
        self.data_bar_stroke_width = data_bar_stroke_width

        self._initializing = False

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    @property
    def columns_show_percentages(self):
        return self._columns_show_percentages

    @columns_show_percentages.setter
    def columns_show_percentages(self, value):
        self._columns_show_percentages = [value] if isinstance(value, str) else value
        # If a provided column name is one of the columns generated here (in particular, df is not None),
        # then we allow the column name to be case insensitive.
        self._columns_show_percentages = [
            GENERATED_COLUMNS.get(col.lower(), col) for col in self._columns_show_percentages
        ]

    @property
    def columns_show_bar(self):
        return self._columns_show_bar

    @columns_show_bar.setter
    def columns_show_bar(self, value):
        self._columns_show_bar = value
        # If a provided column name is one of the columns generated here (in particular, df is not None),
        # then we allow the column name to be case insensitive.
        if self._columns_show_bar:
            self._columns_show_bar = {
                GENERATED_COLUMNS.get(col.lower(), col): self._columns_show_bar[col] for col in self._columns_show_bar
            }


COLORING_CONFIG_DEFAULT = {
    "decimals": 3,
    "columns_show_percentages": ["Prevalence"],
    "percentages_decimals": 2,
}
