from typing import List, Union


class AnalyticsTableConfig:
    """
    Configuration class for table settings in the AnalyticsTable class.
    """

    def __init__(
        self,
        *,
        decimals: int = 3,
        columns_show_percentages: Union[str, List[str]] = "Prevalence",
        percentages_decimals: int = 0,
        alternating_row_colors: bool = True,
        data_bar_stroke_width: int = 4,
    ):
        self.decimals = decimals
        self.columns_show_percentages = columns_show_percentages

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


COLORING_CONFIG_DEFAULT = {
    "decimals": 3,
    "columns_show_percentages": ["Prevalence"],
    "percentages_decimals": 2,
}
