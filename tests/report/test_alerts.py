from unittest.mock import patch

from ydata_profiling.model.alerts import Alert, AlertType

from seismometer.report.alerting import AlertConfigProvider, AlertDef


class Test_Alert_Thresholding:
    @patch.object(AlertConfigProvider, "__init__", return_value=None)
    def test_empty_config(self, mock):
        alert = Alert(AlertType.CONSTANT)

        cfg = AlertConfigProvider()
        cfg._config = AlertDef(**{})

        assert cfg._alert_threshold_met(alert)

    @patch.object(AlertConfigProvider, "__init__", return_value=None)
    def test_empty_config_for_other_alert(self, mock):
        alert = Alert(AlertType.CONSTANT)

        cfg = AlertConfigProvider()
        cfg._config = AlertDef(**{"alerts": {"unique": {"default": {"show": False}}}})

        assert cfg._alert_threshold_met(alert)

    @patch.object(AlertConfigProvider, "__init__", return_value=None)
    def test_not_shown_config(self, mock):
        alert = Alert(AlertType.CONSTANT)

        cfg = AlertConfigProvider()
        cfg._config = AlertDef(**{"alerts": {"constant": {"default": {"show": False}}}})

        assert not cfg._alert_threshold_met(alert)

    @patch.object(AlertConfigProvider, "__init__", return_value=None)
    def test_shown_config(self, mock):
        alert = Alert(AlertType.CONSTANT)

        cfg = AlertConfigProvider()
        cfg._config = AlertDef(**{"alerts": {"constant": {"default": {"show": True}}}})

        assert cfg._alert_threshold_met(alert)

    @patch.object(AlertConfigProvider, "__init__", return_value=None)
    def test_thresholds_not_met_config(self, mock):
        alert = Alert(AlertType.UNIQUE, {"n_unique": 99, "p_unique": 0.04})

        cfg = AlertConfigProvider()
        cfg._config = AlertDef(
            **{"alerts": {"unique": {"default": {"show": True, "thresholds": {"n_unique": 100, "p_unique": 0.05}}}}}
        )

        assert not cfg._alert_threshold_met(alert)

    @patch.object(AlertConfigProvider, "__init__", return_value=None)
    def test_thresholds_one_met_config(self, mock):
        alert = Alert(AlertType.UNIQUE, {"n_unique": 100, "p_unique": 0.04})

        cfg = AlertConfigProvider()
        cfg._config = AlertDef(
            **{"alerts": {"unique": {"default": {"show": True, "thresholds": {"n_unique": 100, "p_unique": 0.05}}}}}
        )

        assert cfg._alert_threshold_met(alert)

    @patch.object(AlertConfigProvider, "__init__", return_value=None)
    def test_thresholds_all_met_config(self, mock):
        alert = Alert(AlertType.UNIQUE, {"n_unique": 100, "p_unique": 0.05})

        cfg = AlertConfigProvider()
        cfg._config = AlertDef(
            **{"alerts": {"unique": {"default": {"show": True, "thresholds": {"n_unique": 100, "p_unique": 0.05}}}}}
        )

        assert cfg._alert_threshold_met(alert)

    @patch.object(AlertConfigProvider, "__init__", return_value=None)
    def test_column_specific_not_shown(self, mock):
        alert = Alert(AlertType.ZEROS, {"n_zeros": 100, "p_zeros": 0.05})
        alert.column_name = "SkipMe!"

        cfg = AlertConfigProvider()
        cfg._config = AlertDef(
            **{
                "alerts": {
                    "zeros": {
                        "default": {"show": True, "thresholds": {"n_zeros": 100, "p_zeros": 0.05}},
                        "columns": {"SkipMe!": {"show": False}},
                    }
                }
            }
        )

        assert not cfg._alert_threshold_met(alert)

    @patch.object(AlertConfigProvider, "__init__", return_value=None)
    def test_column_specific_shown(self, mock):
        alert = Alert(AlertType.ZEROS, {"n_zeros": 100, "p_zeros": 0.05})
        alert.column_name = "DontSkipMe!"

        cfg = AlertConfigProvider()
        cfg._config = AlertDef(
            **{"alerts": {"zeros": {"default": {"show": False}, "columns": {"DontSkipMe!": {"show": True}}}}}
        )

        assert cfg._alert_threshold_met(alert)

    @patch.object(AlertConfigProvider, "__init__", return_value=None)
    def test_column_specific_thresholds_not_met(self, mock):
        alert = Alert(AlertType.ZEROS, {"n_zeros": 100, "p_zeros": 0.05})
        alert.column_name = "SkipMe!"

        cfg = AlertConfigProvider()
        cfg._config = AlertDef(
            **{
                "alerts": {
                    "zeros": {
                        "default": {"show": False},
                        "columns": {"SkipMe!": {"show": True, "thresholds": {"n_zeros": 101, "p_zeros": 0.06}}},
                    }
                }
            }
        )

        assert not cfg._alert_threshold_met(alert)

    @patch.object(AlertConfigProvider, "__init__", return_value=None)
    def test_column_specific_thresholds_met(self, mock):
        alert = Alert(AlertType.ZEROS, {"n_zeros": 100, "p_zeros": 0.05})
        alert.column_name = "DontSkipMe!"

        cfg = AlertConfigProvider()
        cfg._config = AlertDef(
            **{
                "alerts": {
                    "zeros": {
                        "default": {"show": False},
                        "columns": {"DontSkipMe!": {"show": True, "thresholds": {"n_zeros": 100, "p_zeros": 0.05}}},
                    }
                }
            }
        )

        assert cfg._alert_threshold_met(alert)

    @patch.object(AlertConfigProvider, "__init__", return_value=None)
    def test_misconfigured_thresholds_config(self, mock):
        alert = Alert(AlertType.UNIQUE, {"n_unique": 100, "p_unique": 0.05})

        cfg = AlertConfigProvider()
        cfg._config = AlertDef(
            **{"alerts": {"unique": {"default": {"show": True, "thresholds": {"n_dinosaurs": 100}}}}}
        )

        assert not cfg._alert_threshold_met(alert)

    @patch.object(AlertConfigProvider, "__init__", return_value=None)
    def test_misconfigured_thresholds_config_one_met(self, mock):
        alert = Alert(AlertType.UNIQUE, {"n_unique": 100, "p_unique": 0.05})

        cfg = AlertConfigProvider()
        cfg._config = AlertDef(
            **{"alerts": {"unique": {"default": {"show": True, "thresholds": {"n_dinosaurs": 100, "n_unique": 100}}}}}
        )

        assert cfg._alert_threshold_met(alert)

    @patch.object(AlertConfigProvider, "__init__", return_value=None)
    def test_misconfigured_column_specific_thresholds_config(self, mock):
        alert = Alert(AlertType.UNIQUE, {"n_unique": 100, "p_unique": 0.05})
        alert.column_name = "SkipMe!"

        cfg = AlertConfigProvider()
        cfg._config = AlertDef(
            **{
                "alerts": {
                    "unique": {
                        "default": {"show": False},
                        "columns": {"SkipMe!": {"show": True, "thresholds": {"n_dinosaurs": 100}}},
                    }
                }
            }
        )

        assert not cfg._alert_threshold_met(alert)

    @patch.object(AlertConfigProvider, "__init__", return_value=None)
    def test_misconfigured_column_specific_thresholds_config_one_met(self, mock):
        alert = Alert(AlertType.UNIQUE, {"n_unique": 100, "p_unique": 0.05})
        alert.column_name = "SkipMe!"

        cfg = AlertConfigProvider()
        cfg._config = AlertDef(
            **{
                "alerts": {
                    "unique": {
                        "default": {"show": False},
                        "columns": {"SkipMe!": {"show": True, "thresholds": {"n_dinosaurs": 100, "n_unique": 100}}},
                    }
                }
            }
        )

        assert cfg._alert_threshold_met(alert)
