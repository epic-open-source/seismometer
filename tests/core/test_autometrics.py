import logging
from unittest.mock import MagicMock, patch

import pytest
import yaml

from seismometer.core import autometrics
from seismometer.data.performance import BinaryClassifierMetricGenerator


@pytest.fixture(scope="session", autouse=True)
def configure_automation_function_manager():
    autometrics.automation_function_manager = {}


class TestAutometrics:
    def test_get_function(self):
        @autometrics.store_call_parameters
        def foo(a: int, b: int, c: int):
            pass

        assert autometrics.get_function_args("foo") == ["a", "b", "c"]

    def test_transform_item(self):
        assert isinstance(autometrics._transform_item((1, 2, 3)), list)
        assert isinstance(autometrics._transform_item(BinaryClassifierMetricGenerator(rho=0.5)), float)
        assert autometrics._transform_item(2) == 2
        assert autometrics._transform_item("foo") == "foo"

    def test_call_transform(self):
        d_raw = {"foo": 2, "bar": "bar", "baz": (1, 2, 3, 4, 5), "qux": BinaryClassifierMetricGenerator(rho=0.5)}
        d_transformed = autometrics._call_transform(d_raw)
        # Make sure no class tags or anything lying around.
        assert "!!" not in yaml.dump(d_transformed)

    def test_store_call_parameters(self):
        @autometrics.store_call_parameters
        def bar(a: int, b: int, c: int):
            pass

        am = autometrics.AutomationManager()
        bar(1, 2, 3)
        assert am._call_history["bar"] != []

    def test_extract_arguments(self):
        yaml_section = {
            "options": {
                "foo": 1,
                "bar": 2,
                "baz": 3,
                "qux": 4,
            }
        }
        arg_names = ["foo", "bar", "baz"]
        extracted = autometrics.extract_arguments(arg_names, yaml_section)
        for arg in arg_names:
            assert extracted[arg] == yaml_section["options"][arg]
        yaml_section = {"not_options": {"foo": 1, "bar": 2}}
        assert autometrics.extract_arguments(arg_names, yaml_section) == {}

    def test_cohort_function_calls(self):
        variable_dump = []

        def foo(x: int, y: int):
            variable_dump.append(x + y)

        results = {}

        def foo_cohorts(left: int, rights: dict[str, list[int]]):
            for key in rights.keys():
                results[key] = [left + right for right in rights[key]]

        def foo_cohort(left: int, column: str, rights: list[int]):
            results[column] = [left + right for right in rights]

        run_settings = {"options": {"x": 1, "y": 2}}
        autometrics._call_cohortless_function(foo, ["x", "y"], run_settings)
        assert 3 in variable_dump
        variable_dump = []
        run_settings = {
            "options": {
                "left": 1,
            },
            "cohorts": {"foo": [1, 2, 3], "bar": [4, 5, 6]},
        }
        autometrics._call_cohort_dict_function(foo_cohorts, ["left"], run_settings, cohort_dict="rights")
        assert results["foo"] == [2, 3, 4]
        assert results["bar"] == [5, 6, 7]
        run_settings = {"options": {"left": 4}, "cohorts": {"baz": [7, 8, 9], "qux": [10, 11, 12]}}
        autometrics._call_single_cohort_function(
            foo_cohort, ["left"], run_settings, cohort_col="column", subgroups="rights"
        )
        assert results["baz"] == [11, 12, 13]
        assert results["qux"] == [14, 15, 16]

    def test_export_automated_metrics(self):
        mock_am = MagicMock()
        mock_am._automation_info = {"foo": [1, 2, 3], "bar": {4: 5}}
        mock_am.is_allowed_export_function.return_value = True
        with patch("seismometer.core.autometrics.AutomationManager", return_value=mock_am):
            with patch("seismometer.core.autometrics.do_one_export") as mock_do_one_export:
                autometrics.export_automated_metrics()
        print(mock_do_one_export.call_args_list)
        mock_do_one_export.assert_any_call("foo", 1)
        mock_do_one_export.assert_any_call("foo", 2)
        mock_do_one_export.assert_any_call("foo", 3)
        mock_do_one_export.assert_any_call("bar", {4: 5})


# ============================================================================
# ADDITIONAL EDGE CASE TESTS
# ============================================================================


class TestAutomationManagerInitialization:
    """Test AutomationManager initialization with various config scenarios."""

    def test_automation_manager_with_missing_config_files(self):
        """Test AutomationManager handles missing config files gracefully."""
        from pathlib import Path

        mock_config = MagicMock()
        mock_config.automation_config_path = Path("/nonexistent/automation.yml")
        mock_config.automation_config = {}
        mock_config.metric_config = {}

        # Reset singleton instance to allow re-initialization
        autometrics.AutomationManager._instances = {}

        # Should not raise error, just have empty configs
        am = autometrics.AutomationManager(config_provider=mock_config)
        assert am._automation_info == {}
        assert am._metric_info == {}

    def test_automation_manager_with_none_automation_path(self):
        """Test AutomationManager with None automation path."""
        mock_config = MagicMock()
        mock_config.automation_config_path = None
        mock_config.automation_config = {}
        mock_config.metric_config = {}

        # Reset singleton instance
        autometrics.AutomationManager._instances = {}

        am = autometrics.AutomationManager(config_provider=mock_config)
        assert am.automation_file_path is None

    def test_automation_manager_loads_configs_from_provider(self):
        """Test AutomationManager loads configs from ConfigProvider."""
        mock_config = MagicMock()
        mock_config.automation_config_path = "/path/to/automation.yml"
        mock_config.automation_config = {"func1": [{"options": {"a": 1}}]}
        mock_config.metric_config = {"metric1": {"quantiles": 10}}

        # Reset singleton instance
        autometrics.AutomationManager._instances = {}

        am = autometrics.AutomationManager(config_provider=mock_config)
        assert am._automation_info == {"func1": [{"options": {"a": 1}}]}
        assert am._metric_info == {"metric1": {"quantiles": 10}}


class TestStoreCallParametersDecorator:
    """Test store_call_parameters decorator in various scenarios."""

    def test_store_call_parameters_with_positional_args(self):
        """Test decorator stores positional args correctly."""

        @autometrics.store_call_parameters
        def func(a: int, b: int, c: int = 3):
            return a + b + c

        am = autometrics.AutomationManager()
        result = func(1, 2)
        assert result == 6
        assert "func" in am._call_history
        assert am._call_history["func"][-1]["options"] == {"a": 1, "b": 2, "c": 3}

    def test_store_call_parameters_with_kwargs(self):
        """Test decorator stores kwargs correctly."""

        @autometrics.store_call_parameters
        def func(a: int, b: int, c: int = 10):
            return a + b + c

        am = autometrics.AutomationManager()
        result = func(a=5, b=7, c=3)
        assert result == 15
        assert am._call_history["func"][-1]["options"] == {"a": 5, "b": 7, "c": 3}

    def test_store_call_parameters_with_custom_name(self):
        """Test decorator with custom function name."""

        @autometrics.store_call_parameters(name="custom_func_name")
        def internal_func(x: int):
            return x * 2

        am = autometrics.AutomationManager()
        internal_func(5)
        assert "custom_func_name" in am._call_history
        assert "internal_func" not in am._call_history

    def test_store_call_parameters_with_cohort_col(self):
        """Test decorator with cohort_col parameter.

        Note: Cohort parameters appear in both 'options' and 'cohorts' sections.
        This is actual behavior - cohorts are extracted separately for looping purposes.
        """
        # Reset singleton
        autometrics.AutomationManager._instances = {}
        mock_config = MagicMock()
        mock_config.automation_config_path = None
        mock_config.automation_config = {}
        mock_config.metric_config = {}
        am = autometrics.AutomationManager(config_provider=mock_config)

        @autometrics.store_call_parameters(cohort_col="cohort", subgroups="groups")
        def plot_func(cohort: str, groups: list, option: int = 1):
            return f"{cohort}: {groups}"

        plot_func("Age", ["18-25", "26-35"], option=2)

        call_record = am._call_history["plot_func"][-1]
        # Cohorts stored separately for automation looping
        assert call_record["cohorts"] == {"Age": ["18-25", "26-35"]}
        # Options include all parameters (cohorts are not removed)
        assert call_record["options"]["option"] == 2
        assert "cohort" in call_record["options"]
        assert "groups" in call_record["options"]

    def test_store_call_parameters_with_cohort_dict(self):
        """Test decorator with cohort_dict parameter.

        Note: Cohort dict appears in both 'options' and 'cohorts' sections.
        This is actual behavior - cohorts are extracted separately for looping purposes.
        """
        # Reset singleton
        autometrics.AutomationManager._instances = {}
        mock_config = MagicMock()
        mock_config.automation_config_path = None
        mock_config.automation_config = {}
        mock_config.metric_config = {}
        am = autometrics.AutomationManager(config_provider=mock_config)

        @autometrics.store_call_parameters(cohort_dict="cohorts")
        def plot_func(cohorts: dict, option: str = "default"):
            return len(cohorts)

        plot_func({"Age": ["18-25"], "Gender": ["M", "F"]}, option="custom")

        call_record = am._call_history["plot_func"][-1]
        # Cohorts stored separately for automation looping
        assert call_record["cohorts"] == {"Age": ["18-25"], "Gender": ["M", "F"]}
        # Options include all parameters
        assert call_record["options"]["option"] == "custom"
        assert "cohorts" in call_record["options"]

    def test_store_call_parameters_preserves_function_metadata(self):
        """Test decorator preserves function name and docstring."""

        @autometrics.store_call_parameters
        def documented_func(x: int) -> int:
            """This is a docstring."""
            return x + 1

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == "This is a docstring."


class TestGetFunctionArgs:
    """Test get_function_args with edge cases."""

    def test_get_function_args_with_multiple_params(self):
        """Test get_function_args returns all parameter names."""

        @autometrics.store_call_parameters
        def multi_param_func(a: int, b: str, c: float = 1.5, d: bool = False):
            pass

        args = autometrics.get_function_args("multi_param_func")
        assert args == ["a", "b", "c", "d"]

    def test_get_function_args_with_no_params(self):
        """Test get_function_args with function that has no parameters."""

        @autometrics.store_call_parameters
        def no_param_func():
            return 42

        args = autometrics.get_function_args("no_param_func")
        assert args == []


class TestTransformFunctions:
    """Test _transform_item and _call_transform with various types."""

    def test_transform_item_with_list(self):
        """Test _transform_item preserves lists."""
        result = autometrics._transform_item([1, 2, 3])
        assert result == [1, 2, 3]

    def test_transform_item_with_dict(self):
        """Test _transform_item preserves dicts."""
        result = autometrics._transform_item({"a": 1, "b": 2})
        assert result == {"a": 1, "b": 2}

    def test_transform_item_with_none(self):
        """Test _transform_item with None."""
        result = autometrics._transform_item(None)
        assert result is None

    def test_call_transform_with_mixed_types(self):
        """Test _call_transform with mixed value types."""
        data = {
            "int": 42,
            "str": "hello",
            "tuple": (1, 2, 3),
            "none": None,
            "metric_gen": BinaryClassifierMetricGenerator(rho=0.75),
        }
        result = autometrics._call_transform(data)

        assert result["int"] == 42
        assert result["str"] == "hello"
        assert result["tuple"] == [1, 2, 3]  # Tuple converted to list
        assert result["none"] is None
        assert result["metric_gen"] == 0.75  # MetricGenerator converted to rho

    def test_call_transform_preserves_nested_structures(self):
        """Test _call_transform with nested structures."""
        data = {"nested": {"key": (1, 2)}}
        result = autometrics._call_transform(data)

        # Outer dict transformed, but nested dict not recursively transformed
        assert isinstance(result["nested"], dict)
        assert result["nested"]["key"] == (1, 2)  # Inner tuple not transformed


class TestExtractArguments:
    """Test extract_arguments with various YAML structures."""

    def test_extract_arguments_with_missing_options_key(self):
        """Test extract_arguments returns empty dict when 'options' key missing."""
        yaml_section = {"cohorts": {"Age": ["18-25"]}}
        result = autometrics.extract_arguments(["a", "b"], yaml_section)
        assert result == {}

    def test_extract_arguments_with_partial_match(self):
        """Test extract_arguments only extracts args that exist in options."""
        yaml_section = {"options": {"a": 1, "b": 2, "c": 3}}
        result = autometrics.extract_arguments(["a", "d", "e"], yaml_section)
        assert result == {"a": 1}

    def test_extract_arguments_with_no_matches(self):
        """Test extract_arguments when no requested args are in options."""
        yaml_section = {"options": {"x": 10, "y": 20}}
        result = autometrics.extract_arguments(["a", "b"], yaml_section)
        assert result == {}

    def test_extract_arguments_with_empty_options(self):
        """Test extract_arguments with empty options dict."""
        yaml_section = {"options": {}}
        result = autometrics.extract_arguments(["a", "b"], yaml_section)
        assert result == {}


class TestExportConfig:
    """Test export_config functionality."""

    def test_export_config_warning_when_no_path_set(self, caplog):
        """Test export_config logs warning when automation_file_path is None."""
        # Reset singleton
        autometrics.AutomationManager._instances = {}

        mock_config = MagicMock()
        mock_config.automation_config_path = None
        mock_config.automation_config = {}
        mock_config.metric_config = {}

        am = autometrics.AutomationManager(config_provider=mock_config)
        with caplog.at_level(logging.WARNING):
            am.export_config()

        assert "Cannot export config without a file set to export to!" in caplog.text

    def test_export_config_does_not_overwrite_by_default(self, tmp_path):
        """Test export_config does not overwrite existing file by default."""
        # Reset singleton
        autometrics.AutomationManager._instances = {}

        automation_file = tmp_path / "automation.yml"
        automation_file.write_text("existing: content\n")

        mock_config = MagicMock()
        mock_config.automation_config_path = automation_file
        mock_config.automation_config = {}
        mock_config.metric_config = {}

        am = autometrics.AutomationManager(config_provider=mock_config)
        am._call_history = {"func": [{"options": {"new": "data"}}]}

        # Should not overwrite
        am.export_config(overwrite_existing=False)

        content = automation_file.read_text()
        assert "existing: content" in content
        assert "new: data" not in content

    def test_export_config_overwrites_when_requested(self, tmp_path):
        """Test export_config overwrites when overwrite_existing=True."""
        # Reset singleton
        autometrics.AutomationManager._instances = {}

        automation_file = tmp_path / "automation.yml"
        automation_file.write_text("existing: content\n")

        mock_config = MagicMock()
        mock_config.automation_config_path = automation_file
        mock_config.automation_config = {}
        mock_config.metric_config = {}

        am = autometrics.AutomationManager(config_provider=mock_config)
        am._call_history = {"func": [{"options": {"new": "data"}}]}

        # Should overwrite
        am.export_config(overwrite_existing=True)

        content = automation_file.read_text()
        assert "new: data" in content

    def test_export_config_creates_new_file(self, tmp_path):
        """Test export_config creates new file when it doesn't exist."""
        # Reset singleton
        autometrics.AutomationManager._instances = {}

        automation_file = tmp_path / "new_automation.yml"

        mock_config = MagicMock()
        mock_config.automation_config_path = automation_file
        mock_config.automation_config = {}
        mock_config.metric_config = {}

        am = autometrics.AutomationManager(config_provider=mock_config)
        am._call_history = {"test_func": [{"options": {"param": "value"}}]}

        am.export_config(overwrite_existing=False)

        assert automation_file.exists()
        content = yaml.safe_load(automation_file.read_text())
        assert "test_func" in content


class TestGetMetricConfig:
    """Test get_metric_config method."""

    def test_get_metric_config_returns_defaults_for_unknown_metric(self):
        """Test get_metric_config returns defaults when metric not in config."""
        # Reset singleton
        autometrics.AutomationManager._instances = {}

        mock_config = MagicMock()
        mock_config.automation_config_path = None
        mock_config.automation_config = {}
        mock_config.metric_config = {}

        am = autometrics.AutomationManager(config_provider=mock_config)
        config = am.get_metric_config("unknown_metric")

        assert config["output_metrics"] is True
        assert config["log_all"] is False
        assert config["quantiles"] == 4
        assert config["measurement_type"] == "Gauge"

    def test_get_metric_config_merges_with_defaults(self):
        """Test get_metric_config merges custom config with defaults."""
        # Reset singleton
        autometrics.AutomationManager._instances = {}

        mock_config = MagicMock()
        mock_config.automation_config_path = None
        mock_config.automation_config = {}
        mock_config.metric_config = {"my_metric": {"quantiles": 10, "log_all": True}}

        am = autometrics.AutomationManager(config_provider=mock_config)
        config = am.get_metric_config("my_metric")

        # Custom values
        assert config["quantiles"] == 10
        assert config["log_all"] is True
        # Default values
        assert config["output_metrics"] is True
        assert config["measurement_type"] == "Gauge"

    def test_get_metric_config_custom_values_override_defaults(self):
        """Test custom metric config values override defaults."""
        # Reset singleton
        autometrics.AutomationManager._instances = {}

        mock_config = MagicMock()
        mock_config.automation_config_path = None
        mock_config.automation_config = {}
        mock_config.metric_config = {
            "custom_metric": {"output_metrics": False, "quantiles": 20, "measurement_type": "Counter"}
        }

        am = autometrics.AutomationManager(config_provider=mock_config)
        config = am.get_metric_config("custom_metric")

        assert config["output_metrics"] is False
        assert config["quantiles"] == 20
        assert config["measurement_type"] == "Counter"


class TestIsAllowedExportFunction:
    """Test is_allowed_export_function method."""

    def test_is_allowed_export_function_returns_true_for_registered(self):
        """Test is_allowed_export_function returns True for registered functions."""

        @autometrics.store_call_parameters
        def registered_func(x: int):
            return x * 2

        am = autometrics.AutomationManager()
        assert am.is_allowed_export_function("registered_func") is True

    def test_is_allowed_export_function_returns_false_for_unregistered(self):
        """Test is_allowed_export_function returns False for unregistered functions."""
        am = autometrics.AutomationManager()
        assert am.is_allowed_export_function("nonexistent_func") is False


class TestDoExport:
    """Test do_export with different setting types."""

    def test_do_export_with_dict_settings(self):
        """Test do_export handles dict settings."""

        @autometrics.store_call_parameters
        def test_func(a: int, b: int = 2):
            return a + b

        with patch("seismometer.core.autometrics.do_one_export") as mock_do_one:
            autometrics.do_export("test_func", {"options": {"a": 5, "b": 3}})
            mock_do_one.assert_called_once_with("test_func", {"options": {"a": 5, "b": 3}})

    def test_do_export_with_list_settings(self):
        """Test do_export handles list of settings."""

        @autometrics.store_call_parameters
        def test_func(a: int):
            return a

        settings_list = [{"options": {"a": 1}}, {"options": {"a": 2}}, {"options": {"a": 3}}]

        with patch("seismometer.core.autometrics.do_one_export") as mock_do_one:
            autometrics.do_export("test_func", settings_list)
            assert mock_do_one.call_count == 3
            mock_do_one.assert_any_call("test_func", {"options": {"a": 1}})
            mock_do_one.assert_any_call("test_func", {"options": {"a": 2}})
            mock_do_one.assert_any_call("test_func", {"options": {"a": 3}})


class TestExportAutomatedMetrics:
    """Test export_automated_metrics with various scenarios."""

    def test_export_automated_metrics_skips_unrecognized_functions(self, caplog):
        """Test export_automated_metrics logs warning for unrecognized functions."""

        @autometrics.store_call_parameters
        def known_func(x: int):
            return x

        mock_am = MagicMock()
        mock_am._automation_info = {"known_func": {"options": {"x": 1}}, "unknown_func": {"options": {}}}
        mock_am.is_allowed_export_function.side_effect = lambda name: name == "known_func"

        with patch("seismometer.core.autometrics.AutomationManager", return_value=mock_am):
            with patch("seismometer.core.autometrics.do_export") as mock_do_export:
                with patch("seismometer.data.otel.activate_exports"):
                    with caplog.at_level(logging.WARNING):
                        autometrics.export_automated_metrics()

        assert "Unrecognized auto-export function name unknown_func" in caplog.text
        # Only known_func should be exported
        mock_do_export.assert_called_once_with("known_func", {"options": {"x": 1}})


class TestGetFunctionFromExportName:
    """Test get_function_from_export_name with special cases."""

    def test_get_function_from_export_name_for_regular_function(self):
        """Test get_function_from_export_name returns registered function."""

        @autometrics.store_call_parameters
        def my_func(x: int):
            return x * 2

        # Reset singleton
        autometrics.AutomationManager._instances = {}
        mock_config = MagicMock()
        mock_config.automation_config_path = None
        mock_config.automation_config = {}
        mock_config.metric_config = {}
        am = autometrics.AutomationManager(config_provider=mock_config)

        retrieved_fn = am.get_function_from_export_name("my_func")
        # The decorator wraps the function, so compare names instead
        assert retrieved_fn.__name__ == "my_func"
        assert callable(retrieved_fn)
        # Verify it's the same functional behavior
        assert retrieved_fn(5) == 10

    def test_get_function_from_export_name_for_binary_classifier_metrics(self):
        """Test special case for plot_binary_classifier_metrics."""
        am = autometrics.AutomationManager()

        # Special case that imports from seismometer.api.plots
        with patch("seismometer.api.plots._autometric_plot_binary_classifier_metrics") as mock_func:
            retrieved_fn = am.get_function_from_export_name("plot_binary_classifier_metrics")
            assert retrieved_fn == mock_func

    def test_get_function_from_export_name_for_fairness_table(self):
        """Test special case for binary_metrics_fairness_table."""
        am = autometrics.AutomationManager()

        # Special case that imports from seismometer.table.fairness
        with patch("seismometer.table.fairness._autometric_plot_binary_classifier_metrics") as mock_func:
            retrieved_fn = am.get_function_from_export_name("binary_metrics_fairness_table")
            assert retrieved_fn == mock_func
