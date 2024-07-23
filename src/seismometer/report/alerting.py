""" example alert_config.yaml
alerts:
  constant:
    default:
      show: true
    columns:
      "Absolute_Lymphocytes":
        show: false
      "Relative_Lymphocytes":
        show: true

  unique:
    default:
      show: true
      thresholds:
        n_unique: 100
        p_unique: 0.05
    columns:
      "Absolute_Lymphocytes":
        show: true
        thresholds:
          n_unique: 1000
          p_unique: 0.5
      "Relative_Lymphocytes":
        show: false

  imbalance:
    default:
      show: true
"""

import logging
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

from seismometer.plot.mpl._ux import alert_colors

logger = logging.getLogger("seismometer")


class ParsedAlert(BaseModel):
    name: str
    severity: str
    display_html: str


class ParsedAlertList(BaseModel):
    alerts: list[ParsedAlert]


class AlertThresholds(BaseModel):
    show: bool = True
    thresholds: Optional[dict[str, float]] = {}


class Alert(BaseModel):
    default: Optional[AlertThresholds] = None
    columns: Optional[dict[str, AlertThresholds]] = {}
    severity: Optional[str] = None


class AlertDef(BaseModel):
    alerts: Optional[dict[str, Alert]] = {}


class AlertConfigProvider:
    _config: AlertDef

    def __init__(self, config_file_path: str | Path) -> None:
        config_path = Path(config_file_path)
        if config_path.is_dir():
            self.config_dir: Path = config_path
            self.config_file: str = "alert_config.yml"
        else:
            self.config_dir: Path = config_path.parent
            self.config_file: str = config_path.name
        if not (self.config_dir / self.config_file).is_file():
            logger.debug(f"No config file found at {(self.config_dir / self.config_file).resolve()}")
            self._config = AlertDef()
        else:
            with open(self.config_dir / self.config_file, "r") as f:
                self._config = AlertDef(**yaml.safe_load(f))

    def parse_alert(self, alert: "ydata_profiling.model.alerts.Alert") -> tuple[bool, str]:
        """
        Parses a single `ydata-profiling` Alert object.

        Parameters
        ----------
        alert : ydata_profiling.model.alerts.Alert
            The alert to parse.

        Returns
        -------
        tuple[bool, str]
            bool: True if the alert met the threshold, otherwise False.
            str: The semantic color indicating the severity of the alert.
        """

        return self._alert_threshold_met(alert), self._alert_style(alert)

    def _alert_style(self, alert: "ydata_profiling.model.alerts.Alert") -> str:
        """
        Returns the configured severity for an alert, or the default specified here.

        Parameters
        ----------
        alert : ydata_profiling.model.alerts.Alert
            The alert to return data for.

        Returns
        -------
        str
            A hex representation of the color / severity.
        """
        styles = {
            "constant": alert_colors.Important,
            "unsupported": alert_colors.Important,
            "type_date": alert_colors.Important,
            "high_cardinality": alert_colors.Alarm,
            "unique": alert_colors.Alarm,
            "uniform": alert_colors.Alarm,
            "infinite": alert_colors.Warning,
            "zeros": alert_colors.Warning,
            "missing": alert_colors.Warning,
            "skewed": alert_colors.Warning,
            "imbalance": alert_colors.Warning,
            "high_correlation": alert_colors.Normal,
            "duplicates": alert_colors.Normal,
            "empty": alert_colors.Normal,
            "non_stationary": alert_colors.Normal,
            "seasonal": alert_colors.Normal,
        }

        if (
            not self._config.alerts
            or not (config := self._config.alerts.get(alert.alert_type.name.lower()))
            or not config.severity
        ):
            return styles[alert.alert_type.name.lower()]

        return config.severity

    def _alert_threshold_met(self, alert: "ydata_profiling.model.alerts.Alert") -> bool:
        """
        Determines if an Alert meets its configured thresholds.

        Parameters
        ----------
        alert : ydata_profiling.model.alerts.Alert
            The alert to check.

        Returns
        -------
        bool
            True if
                1) there is no thresholds configured for the alert (or generally),

                2) _alert_threshold_met_col returns True for the alert.
        """

        if not self._config.alerts:
            return True  # no config, show by default

        if not (config := self._config.alerts.get(alert.alert_type.name.lower())):
            return True  # no alert-specific config, show by default

        if column_config := config.columns.get(alert.column_name):
            return self._alert_threshold_met_col(alert, column_config)  # show if matches column-specific config
        else:
            return self._alert_threshold_met_col(alert, config.default)  # show if matches default config

    def _alert_threshold_met_col(self, alert: "ydata_profiling.model.alerts.Alert", col_cfg: AlertThresholds) -> bool:
        """
        Helper API to determine if threshold is met for specific alert and specific alert threshold.

        Parameters
        ----------
        alert : ydata_profiling.model.alerts.Alert
            The `ydata-profiling` alert object.
        col_cfg : AlertThresholds
            The alert threshold object created from config.

        Returns
        -------
        bool
            True if
                1) the alert was set to show, and was configured with no thresholds, or

                2) the alert was set to show, was configured with thresholds, and met or exceeded at least one.
            False otherwise
        """
        if not col_cfg:
            return True

        if not col_cfg.show:
            return False

        if not col_cfg.thresholds:
            return True  # vacuously meets thresholds

        for threshold in col_cfg.thresholds.keys():
            if threshold not in alert.values:
                logger.warning(f"Misconfigured alert {threshold=} for {alert}.")
            elif alert.values[threshold] >= col_cfg.thresholds[threshold]:
                return True  # at least one threshold matched

        return False  # no thresholds matched
