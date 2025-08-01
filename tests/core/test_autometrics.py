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
