import json
import logging
import os
import re
from abc import ABC
from importlib.resources import files as _files
from pathlib import Path

from IPython.display import HTML as IPyHTML
from IPython.display import IFrame, display
from ipywidgets import HTML, Button, ButtonStyle, GridBox, Layout
from pandas import DataFrame
from seaborn.utils import relative_luminance

import seismometer.report
from seismometer.core.io import slugify

from .alerting import AlertConfigProvider, ParsedAlert, ParsedAlertList

PROFILING_CONFIG_PATH = _files(seismometer.report) / "report_config.yml"
logger = logging.getLogger("seismometer")


def filter_unsupported_columns(df: DataFrame) -> DataFrame:
    """
    Filters out columns that are not supported by ydata-profiling.

    Parameters
    ----------
    df : DataFrame
        A Pandas DataFrame object.

    Returns
    -------
    DataFrame
        A Pandas DataFrame object with unsupported columns removed.
    """
    return df.select_dtypes(exclude=["datetime", "datetimetz", "datetime64[ns]"])


class ReportWrapper(ABC):
    _df: DataFrame
    _html_path: str
    _pickle_path: str
    _report: "ProfileReport"
    _title: str

    def display_report(self, inline: bool) -> None:
        """
        Displays the HTML ProfileReport associated with the wrapper.

        Parameters
        ----------
        inline : bool
            If true, will display the ProfileReport in an iframe, else shows a link.
        """
        if inline:
            display(IFrame(src=os.path.relpath(self._html_path), width="100%", height="800px"))
        else:
            display(IPyHTML(f'<a href="{os.path.relpath(self._html_path)}" target="_blank">Open {self._title}</a>'))

    def save_report(self):
        self._report.to_file(self._html_path)


class ComparisonReportWrapper(ReportWrapper):
    def __init__(
        self,
        l_df: DataFrame,
        r_df: DataFrame,
        output_path: Path,
        l_title: str = "Left",
        r_title: str = "Right",
        *,
        exclude_cols: list[str] = None,
        base_title: str = "Feature Comparison Report",
    ) -> None:
        """
        Creates a wrapper around a `ydata-profiling` comparison ProfileReport.

        Splits the `df` by `compare_col`, filters to the `first_groups` and `second_groups`,
        then compares the two datasets.
        If an HTML report exists with the same name, it will not re-generate the report.

        Parameters
        ----------
        df : DataFrame
            A Pandas DataFrame object.
        output_path: Path
            Location to store the report.
        l_compare_col : str
            The column name for the first group.
        r_compare_col : list[str]
            The column name for the second group.
        l_groups : list[str]
            Subgroups within the first group.
        r_groups : list[str]
            Subgroups within the second group.
        exclude_cols : Optional[list[str]], optional
            Columns to exclude from profiling, by default None.
        base_title : str, optional
            Base title for the report, by default "Feature Report".
        """

        self._l_df = filter_unsupported_columns(l_df)
        self._r_df = filter_unsupported_columns(r_df)
        self._l_title = l_title
        self._r_title = r_title
        self._exclude_cols = exclude_cols

        self._title = f"{base_title} Comparing {l_title} against {r_title}"

        self._html_path = output_path / (slugify(self._title) + ".html")

        super().__init__()

        if not Path(self._html_path).is_file():
            logger.debug(f"Generating and saving report: {self._html_path}")
            self.generate_report()
            self.save_report()
        else:
            logger.debug(f"Existing report found on disk: {self._html_path}")

    def generate_report(self):
        try:
            from ydata_profiling import ProfileReport
        except ImportError:
            raise ImportError(
                "Error: ydata-profiling or one of its required packages is not installed. Install with "
                "`pip install ydata-profiling`."
            )

        dfs = [self._l_df, self._r_df]
        if self._exclude_cols is not None:
            dfs = [df.loc[:, [col for col in df.columns if col not in self._exclude_cols]] for df in dfs]

        first_report = ProfileReport(dfs[0], config_file=PROFILING_CONFIG_PATH, title=self._l_title)
        second_report = ProfileReport(dfs[1], config_file=PROFILING_CONFIG_PATH, title=self._r_title)

        self._report = first_report.compare(second_report)


class SingleReportWrapper(ReportWrapper):
    _parsed_alerts: ParsedAlertList

    def __init__(
        self,
        df: DataFrame,
        output_path: Path,
        exclude_cols: list[str] = None,
        title: str = "Report",
        alert_config: AlertConfigProvider = None,
    ):
        """
        Creates a wrapper around a `ydata-profiling` ProfileReport.
        If an HTML report exists with the same name, it will not re-generate the report.

        Parameters
        ----------
        df : DataFrame
            A pandas DataFrame object.
        exclude_cols : Optional[list[str]], optional
            Columns to exclude from profiling, by default None.
        title : str, optional
            Title for the report, by default "Report".
        alert_config : Optional[AlertConfigProvider], optional
            The parsed configuration for determining which alerts are shown, by default None.
        """
        self._df = filter_unsupported_columns(df)
        self._title = title
        self._html_path = output_path / (slugify(title) + ".html")
        self._alert_path = output_path / (slugify(title) + "_alerts.json")
        self._exclude_cols = exclude_cols
        self._alert_config = alert_config

        super().__init__()

        if not Path(self._html_path).is_file() or not Path(self._alert_path).is_file():
            logger.debug(f"Generating and saving report: {self._html_path}")
            self.generate_report()
            self.save_report()

            logger.debug(f"Parsing and saving alerts: {self._alert_path}")
            self._parse_alerts()
            self._serialize_alerts()
        else:
            logger.debug(f"Existing report found on disk: {self._html_path}")

            logger.debug(f"Existing alerts found on disk: {self._alert_path}")
            self._deserialize_alerts()

    def generate_report(self) -> None:
        try:
            from ydata_profiling import ProfileReport
        except ImportError:
            raise ImportError(
                "Error: ydata-profiling or one of its required packages is not installed. "
                "Install with `pip install ydata-profiling`."
            )

        self._df = self._df.loc[:, [col for col in self._df.columns if col not in self._exclude_cols]]
        self._report = ProfileReport(self._df, title=self._title, config_file=PROFILING_CONFIG_PATH)

    def _parse_alerts(self) -> None:
        """
        Parses all of the alerts associated with the `ydata-profiling` ProfileReport.
        """
        alerts = []
        if len(self._report.description_set.alerts) == 0:
            self._parsed_alerts = ParsedAlertList(alerts=alerts)
            return

        # else, parse raw alerts and template html
        try:
            from ydata_profiling.model.alerts import Alert
            from ydata_profiling.report.presentation.flavours.html import templates
        except ImportError:
            raise ImportError(
                "Error: ydata-profiling or one of its required packages is not installed. "
                "Install with `pip install ydata-profiling`."
            )

        alert: Alert
        for alert in self._report.description_set.alerts:
            if alert.alert_type.name.lower() == "rejected":
                continue  # rejected variables don't have alerts in ydata
            threshold_met, severity = self._alert_config.parse_alert(alert)
            if threshold_met:
                alerts.append(
                    ParsedAlert(
                        name=alert.alert_type_name,
                        severity=severity,
                        display_html=str(
                            templates.template(f"alerts/alert_{alert.alert_type.name.lower()}.html").render(
                                alert=alert
                            )
                        ),
                    )
                )

        self._parsed_alerts = ParsedAlertList(alerts=alerts)
        self._serialize_alerts()

    def _serialize_alerts(self) -> None:
        with open(self._alert_path, "w") as f:
            f.write(self._parsed_alerts.model_dump_json())

    def _deserialize_alerts(self) -> None:
        with open(self._alert_path, "r") as f:
            self._parsed_alerts = ParsedAlertList(**json.load(f))

    def display_alerts(self) -> None:
        """
        Returns the alerts produced by the `ydata-profiling` ProfileReport.
        """
        if len(self._parsed_alerts.alerts) == 0:
            display(HTML("<p>No alerts to display</p>"))

        html_children = []

        alert: ParsedAlert
        for alert in self._parsed_alerts.alerts:
            html_children.append(
                Button(
                    description=alert.name,
                    disabled=True,
                    style=ButtonStyle(
                        button_color=alert.severity,
                        text_color="#000000" if relative_luminance(alert.severity) > 0.4 else "#ffffff",
                    ),
                    layout=Layout(width="95%"),
                )
            )
            html_children.append(HTML(re.sub(r"<a[^>]*>(.*?)</a>", r"\1", alert.display_html)))

        display(
            HTML(
                "<style>.widget-html-content > code { "
                + "padding:2px 4px;font-size:90%;color:#c7254e;background-color:#f9f2f4;border-radius:4px } "
                + ":root { --jp-widgets-disabled-opacity: 0.8; }</style>"
            )
        )

        display(GridBox(children=html_children, layout=Layout(grid_template_columns="10% 90%", width="100%")))
