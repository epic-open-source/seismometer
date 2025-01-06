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
    ):
        """
        Initializes the AnalyticsTableConfig class with provided parameters.

        Parameters
        ----------
        decimals : int, optional
            The number of decimal places for rounding numerical results, by default 3.
        columns_show_percentages : Union[str, List[str]], optional
            Columns that will display values as percentages in the analytics table, by default "Prevalence".
        percentages_decimals : int, optional
            The number of decimal places for percentage values in the analytics table, by default 0.
        """

        self.decimals = decimals
        self.columns_show_percentages = columns_show_percentages

        self.percentages_decimals = percentages_decimals

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
