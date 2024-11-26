from enum import Enum
from typing import Dict, List, Union

from matplotlib.colors import is_color_like

GENERATED_COLUMNS = {
    "positives": "Positives",
    "prevalence": "Prevalence",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "accuracy": "Accuracy",
    "ppv": "PPV",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "flagged": "Flag Rate",
    "threshold": "Threshold",
}


class Metric(Enum):
    """
    Enumeration for available values for metric parameter in PerformanceMetrics class.
    """

    Sensitivity = "sensitivity"
    Specificity = "specificity"
    Flagged = "flagged"
    Threshold = "threshold"


class TopLevel(Enum):
    """
    Enumeration for available values for top_level parameter in PerformanceMetrics class.
    """

    Score = "score"
    Target = "target"


class ColorBarStyle(Enum):
    """
    Enumeration for different color bar styles available in PerformanceMetrics class.
    """

    Style1 = 1  # Value and color bar in two adjacent columns
    Style2 = 2  # Value behind color bar


class GTStyle(Enum):
    """
    Enumeration for different table styles available in great_tables package.
    """

    Style1 = 1
    Style2 = 2
    Style3 = 3
    Style4 = 4
    Style5 = 5
    Style6 = 6


class AnalyticsTableConfig:
    """
    Configuration class for table settings in the PerformanceMetrics class.
    """

    def __init__(
        self,
        *,
        decimals: int = 3,
        spanner_colors: List[str] = None,
        columns_show_percentages: Union[str, List[str]] = "Prevalence",
        columns_show_bar: Dict[str, str] = None,
        color_bar_style: int = 1,
        style: int = 1,
        opacity: int = 0.5,
        percentages_decimals: int = 0,
        alternating_row_colors: bool = True,
        data_bar_stroke_width: int = 4,
    ):
        self.decimals = decimals
        self.spanner_colors = spanner_colors
        self.columns_show_percentages = columns_show_percentages
        self.columns_show_bar = columns_show_bar

        self.color_bar_style = ColorBarStyle(color_bar_style).value
        self.opacity = opacity
        self.style = GTStyle(style).value
        self.percentages_decimals = percentages_decimals
        self.alternating_row_colors = alternating_row_colors
        self.data_bar_stroke_width = data_bar_stroke_width

        self._initializing = False

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

    @property
    def spanner_colors(self):
        return self._spanner_colors

    @spanner_colors.setter
    def spanner_colors(self, value):
        self._spanner_colors = value if value else []
        for color in self._spanner_colors:
            if not is_color_like(color):
                raise ValueError(f"Invalid color: {color}")

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
        self._columns_show_bar = value if value else {"AUROC": "lightblue", "PPV": "lightgreen"}
        # If a provided column name is one of the columns generated here (in particular, df is not None),
        # then we allow the column name to be case insensitive.
        self._columns_show_bar = {
            GENERATED_COLUMNS.get(col.lower(), col): self._columns_show_bar[col] for col in self._columns_show_bar
        }

    @property
    def color_bar_style(self):
        return self._color_bar_style

    @color_bar_style.setter
    def color_bar_style(self, value):
        self._color_bar_style = ColorBarStyle(value).value

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, value):
        self._style = GTStyle(value).value


COLORING_SCHEMA_1 = {
    "decimals": 3,
    "columns_show_percentages": ["Prevalence"],
    "columns_show_bar": {"AUROC": "lightblue", "PPV": "lightblue"},
    "color_bar_style": 1,
    "style": 1,
    "opacity": 0.5,
    "percentages_decimals": 2,
}

COLORING_SCHEMA_2 = {
    "decimals": 3,
    "spanner_colors": ["red", "blue"],
    "columns_show_percentages": ["Prevalence"],
    "columns_show_bar": {"Prevalence": "bar"},
    "color_bar_style": 2,
    "style": 2,
    "opacity": 0.7,
    "percentages_decimals": 1,
    "alternating_row_colors": False,
    "data_bar_stroke_width": 5,
}

COLORING_SCHEMA_3 = {
    "decimals": 3,
    "spanner_colors": ["red", "blue"],
    "columns_show_percentages": ["Prevalence", "Accuracy"],
    "columns_show_bar": {"Prevalence": "bar"},
    "color_bar_style": 2,
    "style": 2,
    "opacity": 0.7,
    "percentages_decimals": 1,
    "alternating_row_colors": False,
    "data_bar_stroke_width": 5,
}


class AnalyticsTableStyle(Enum):
    Style1 = 1
    Style2 = 2
    Style3 = 3

    @staticmethod
    def get_style(style_number):
        if style_number == 1:
            return COLORING_SCHEMA_1
        elif style_number == 2:
            return COLORING_SCHEMA_2
        elif style_number == 3:
            return COLORING_SCHEMA_3
        else:
            raise ValueError("Invalid style number. Choose 1, 2, or 3.")
