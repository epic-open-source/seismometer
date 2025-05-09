import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml
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

    def test_alert_config_file_loaded_from_file(self):
        cfg = {"alerts": {"constant": {"severity": "#123456"}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "alert_config.yml"
            with open(config_path, "w") as f:
                yaml.dump(cfg, f)

            provider = AlertConfigProvider(config_path)
            assert provider._config.alerts["constant"].severity == "#123456"

    def test_alert_config_file_not_found_falls_back(self, caplog):
        with tempfile.TemporaryDirectory() as tmpdir:
            with caplog.at_level("DEBUG"):
                provider = AlertConfigProvider(Path(tmpdir) / "missing_config.yml")
                assert provider._config.alerts == {}
                assert "No config file found" in caplog.text

    def test_alert_config_file_from_directory_path(self, tmp_path):
        config = {"alerts": {"constant": {"severity": "#abcdef"}}}
        config_file = tmp_path / "alert_config.yml"
        config_file.write_text(yaml.dump(config))
        provider = AlertConfigProvider(tmp_path)
        assert provider._config.alerts["constant"].severity == "#abcdef"

    def test_alert_style_default_and_custom(self):
        alert = Alert(AlertType.CONSTANT)
        provider = AlertConfigProvider.__new__(AlertConfigProvider)
        provider._config = type(
            "MockConfig", (), {"alerts": {"constant": type("MockAlert", (), {"severity": "#ffff00"})()}}
        )
        assert provider._alert_style(alert) == "#ffff00"

        alert = Alert(AlertType.MISSING)
        provider._config = type("MockConfig", (), {"alerts": {}})
        from seismometer.plot.mpl._ux import alert_colors

        assert provider._alert_style(alert) == alert_colors.Warning
