from unittest.mock import MagicMock, Mock, patch

import ipywidgets
import pytest

import seismometer.controls.explore as undertest
from seismometer import seismogram
from seismometer.controls.categorical import CategoricalOptionsWidget, ExploreCategoricalPlots
from seismometer.controls.categorical_single_column import (
    CategoricalFeedbackSingleColumnOptionsWidget,
    ExploreSingleCategoricalPlots,
)
from seismometer.data.performance import MetricGenerator
from seismometer.table.analytics_table import AnalyticsTableOptionsWidget, ExploreBinaryModelAnalytics


# region Test Base Classes
class TestUpdatePlotWidget:
    def test_init(self):
        widget = undertest.UpdatePlotWidget()
        assert widget.plot_button.description == undertest.UpdatePlotWidget.UPDATE_PLOTS
        assert widget.plot_button.disabled is False
        assert widget.code_checkbox.description == widget.SHOW_CODE
        assert widget.show_code is False

    def test_plot_button_click(self):
        count = 0

        def on_click_callback(button):
            nonlocal count
            count += 1

        widget = undertest.UpdatePlotWidget()
        widget.on_click(on_click_callback)
        widget.plot_button.click()
        assert count == 1

    def test_disable(self):
        widget = undertest.UpdatePlotWidget()
        widget.disabled = True
        assert widget.plot_button.disabled
        assert widget.disabled

    def test_enable(self):
        widget = undertest.UpdatePlotWidget()
        widget.disabled = True
        widget.disabled = False
        assert not widget.plot_button.disabled
        assert not widget.disabled
        assert widget.plot_button.description == undertest.UpdatePlotWidget.UPDATE_PLOTS

    def test_toggle_code_checkbox(self):
        count = 0

        def on_toggle_callback(button):
            nonlocal count
            count += 1

        widget = undertest.UpdatePlotWidget()
        widget.on_toggle_code(on_toggle_callback)
        widget.code_checkbox.value = True
        assert count == 1


class TestExplorationBaseClass:
    def test_base_class(self, caplog):
        option_widget = ipywidgets.Checkbox(description="ClickMe")
        plot_function = Mock(return_value="some result")
        widget = undertest.ExplorationWidget("ExploreTest", option_widget, plot_function)

        plot_function.assert_not_called()
        assert "Subclasses must implement this method" in widget.center.outputs[0]["data"]["text/plain"]

    @pytest.mark.parametrize(
        "plot_module,plot_code",
        [
            ("__main__", "plot_something(False)"),
            ("seismometer.api", "sm.plot_something(False)"),
            ("something_else", "something_else.plot_something(False)"),
        ],
    )
    def test_args_subclass(self, plot_module, plot_code):
        option_widget = ipywidgets.Checkbox(description="ClickMe")
        plot_function = Mock(return_value="some result")
        plot_function.__name__ = "plot_something"
        plot_function.__module__ = plot_module

        class ExploreFake(undertest.ExplorationWidget):
            def __init__(self):
                super().__init__("Fake Explorer", option_widget, plot_function)

            def generate_plot_args(self) -> tuple[tuple, dict]:
                return [self.option_widget.value], {}

        widget = ExploreFake()
        assert widget.center.outputs[0]["data"]["text/plain"] == "'some result'"
        plot_function.assert_called_once_with(False)
        assert widget.current_plot_code == plot_code
        assert widget.show_code is False

    def test_kwargs_subclass(self):
        option_widget = ipywidgets.Checkbox(description="ClickMe")
        plot_function = Mock(return_value="some result")
        plot_function.__name__ = "plot_something"
        plot_function.__module__ = "test_explore"

        class ExploreFake(undertest.ExplorationWidget):
            def __init__(self):
                super().__init__("Fake Explorer", option_widget, plot_function)

            def generate_plot_args(self) -> tuple[tuple, dict]:
                return [], {"checkbox": self.option_widget.value}

        widget = ExploreFake()
        assert widget.center.outputs[0]["data"]["text/plain"] == "'some result'"
        plot_function.assert_called_once_with(checkbox=False)
        assert widget.current_plot_code == "test_explore.plot_something(checkbox=False)"
        assert widget.show_code is False

    def test_args_kwargs_subclass(self):
        option_widget = ipywidgets.Checkbox(description="ClickMe")
        plot_function = Mock(return_value="some result")
        plot_function.__name__ = "plot_something"
        plot_function.__module__ = "test_explore"

        class ExploreFake(undertest.ExplorationWidget):
            def __init__(self):
                super().__init__("Fake Explorer", option_widget, plot_function)

            def generate_plot_args(self) -> tuple[tuple, dict]:
                return ["test"], {"checkbox": self.option_widget.value}

        widget = ExploreFake()
        assert widget.center.outputs[0]["data"]["text/plain"] == "'some result'"
        plot_function.assert_called_once_with("test", checkbox=False)
        assert widget.current_plot_code == "test_explore.plot_something('test', checkbox=False)"
        assert widget.show_code is False

    def test_exception_plot_code_subclass(self):
        option_widget = ipywidgets.Checkbox(description="ClickMe")

        def plot_something(*args, **kwargs):
            raise ValueError("Test Exception")

        class ExploreFake(undertest.ExplorationWidget):
            def __init__(self):
                super().__init__("Fake Explorer", option_widget, plot_something)

            def generate_plot_args(self) -> tuple[tuple, dict]:
                return ["test"], {"checkbox": self.option_widget.value}

        widget = ExploreFake()
        assert "Traceback" in widget.center.outputs[0]["data"]["text/plain"]
        assert "Test Exception" in widget.center.outputs[0]["data"]["text/plain"]

    def test_no_initial_plot_subclass(self):
        option_widget = ipywidgets.Checkbox(description="ClickMe")
        plot_function = Mock(return_value="some result")
        plot_function.__name__ = "plot_something"
        plot_function.__module__ = "test_explore"

        class ExploreFake(undertest.ExplorationWidget):
            def __init__(self):
                super().__init__("Fake Explorer", option_widget, plot_function, initial_plot=False)

            def generate_plot_args(self) -> tuple[tuple, dict]:
                return ["test"], {"checkbox": self.option_widget.value}

        widget = ExploreFake()
        widget.center.outputs == []
        plot_function.assert_not_called()
        assert widget.current_plot_code == ExploreFake.NO_CODE_STRING
        assert widget.show_code is False

    @pytest.mark.parametrize("show_code", [True, False])
    def test_toggle_code_callback(self, show_code, capsys):
        option_widget = ipywidgets.Checkbox(description="ClickMe")
        plot_function = Mock(return_value="some result")
        plot_function.__name__ = "plot_something"
        plot_function.__module__ = "test_explore"

        class ExploreFake(undertest.ExplorationWidget):
            def __init__(self):
                super().__init__("Fake Explorer", option_widget, plot_function)

            def generate_plot_args(self) -> tuple[tuple, dict]:
                return ["test"], {"checkbox": self.option_widget.value}

        widget = ExploreFake()
        widget.show_code = show_code
        widget.code_output = MagicMock()
        widget._on_plot_button_click()
        stdout = capsys.readouterr().out
        assert "some result" in stdout
        code_in_output = "test_explore.plot_something" in stdout.split("\n")[-2]
        assert code_in_output == show_code


# endregion
# region Test Model Options Widgets


class TestModelOptionsWidget:
    def test_init_with_all_options(self):
        widget = undertest.ModelOptionsWidget(
            target_names=["T1", "T2"], score_names=["S1", "S2"], thresholds={"T1": 0.2, "T2": 0.1}, per_context=True
        )
        assert len(widget.children) == 5
        assert "Model Options" in widget.title.value
        assert widget.target == "T1"
        assert widget.score == "S1"
        assert widget.thresholds == {"T1": 0.2, "T2": 0.1}
        assert widget.group_scores is True

    def test_init_with_all_options_grouping_off(self):
        widget = undertest.ModelOptionsWidget(
            target_names=["T1", "T2"], score_names=["S1", "S2"], thresholds={"T1": 0.2, "T2": 0.1}, per_context=False
        )
        assert len(widget.children) == 5
        assert "Model Options" in widget.title.value
        assert widget.target == "T1"
        assert widget.score == "S1"
        assert widget.thresholds == {"T1": 0.2, "T2": 0.1}
        assert widget.group_scores is False

    def test_no_combine_scores_checkbox(self):
        widget = undertest.ModelOptionsWidget(
            target_names=["T1", "T2"], score_names=["S1", "S2"], thresholds={"T1": 0.1, "T2": 0.2}
        )
        assert len(widget.children) == 4
        assert "Model Options" in widget.title.value
        assert widget.target == "T1"
        assert widget.score == "S1"
        assert widget.thresholds == {"T1": 0.1, "T2": 0.2}
        assert widget.per_context_checkbox is None

    def test_no_score_thresholds(self):
        widget = undertest.ModelOptionsWidget(target_names=["T1", "T2"], score_names=["S1", "S2"])
        assert len(widget.children) == 3
        assert "Model Options" in widget.title.value
        assert widget.target == "T1"
        assert widget.score == "S1"
        assert widget.threshold_list is None
        assert widget.per_context_checkbox is None

    def test_per_context_checkbox(self):
        widget = undertest.ModelOptionsWidget(target_names=["T1", "T2"], score_names=["S1", "S2"], per_context=False)

        widget.per_context_checkbox.value = True
        assert widget.group_scores

    def test_disabled(self):
        widget = undertest.ModelOptionsWidget(
            target_names=["T1", "T2"], score_names=["S1", "S2"], thresholds={"T1": 0.1, "T2": 0.2}, per_context=False
        )
        widget.disabled = True
        assert widget.target_list.disabled
        assert widget.score_list.disabled
        assert widget.threshold_list.disabled
        assert widget.per_context_checkbox.disabled
        assert widget.disabled

    def test_disabled_by_list_size(self):
        widget = undertest.ModelOptionsWidget(
            target_names=["T1"], score_names=["S1"], thresholds={"T1": 0.1, "T2": 0.2}, per_context=False
        )
        widget.disabled = False
        assert widget.target_list.disabled
        assert widget.score_list.disabled
        assert not widget.threshold_list.disabled
        assert not widget.per_context_checkbox.disabled
        assert not widget.disabled


class TestModelOptionsAndCohortsWidget:
    def test_init(self):
        widget = undertest.ModelOptionsAndCohortsWidget(
            cohort_groups={"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]},
            target_names=["T1", "T2"],
            score_names=["S1", "S2"],
            thresholds={"T1": 0.2, "T2": 0.1},
            per_context=True,
        )
        assert len(widget.children) == 2
        assert widget.cohorts == {}
        assert widget.target == "T1"
        assert widget.score == "S1"
        assert widget.thresholds == {"T1": 0.2, "T2": 0.1}
        assert widget.group_scores is True

    def test_disable(self):
        widget = undertest.ModelOptionsAndCohortsWidget(
            cohort_groups={"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]},
            target_names=["T1", "T2"],
            score_names=["S1", "S2"],
            thresholds={"T1": 0.2, "T2": 0.1},
            per_context=True,
        )
        widget.disabled = True
        assert widget.model_options.disabled
        assert widget.cohort_list.disabled
        assert widget.disabled


class TestModelOptionsAndCohortGroupWidget:
    def test_init(self):
        widget = undertest.ModelOptionsAndCohortGroupWidget(
            cohort_groups={"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]},
            target_names=["T1", "T2"],
            score_names=["S1", "S2"],
            thresholds={"T1": 0.2, "T2": 0.1},
            per_context=True,
        )
        assert len(widget.children) == 2
        assert widget.cohort == "C1"
        assert widget.cohort_groups == ("C1.1", "C1.2")
        assert widget.target == "T1"
        assert widget.score == "S1"
        assert widget.thresholds == {"T1": 0.2, "T2": 0.1}
        assert widget.group_scores is True

    def test_disable(self):
        widget = undertest.ModelOptionsAndCohortGroupWidget(
            cohort_groups={"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]},
            target_names=["T1", "T2"],
            score_names=["S1", "S2"],
            thresholds={"T1": 0.2, "T2": 0.1},
            per_context=True,
        )
        widget.disabled = True
        assert widget.model_options.disabled
        assert widget.cohort_list.disabled
        assert widget.disabled


class TestModelInterventionOptionsWidget:
    def test_init(self):
        widget = undertest.ModelInterventionOptionsWidget(
            outcome_names=["O1", "O2"], intervention_names=["I1", "I2"], reference_time_names=["R1", "R2"]
        )

        assert len(widget.children) == 4
        assert "Model Options" in widget.title.value
        assert widget.outcome == "O1"
        assert widget.intervention == "I1"
        assert widget.reference_time == "R1"

    def test_disable(self):
        widget = undertest.ModelInterventionOptionsWidget(
            outcome_names=["O1", "O2"], intervention_names=["I1", "I2"], reference_time_names=["R1", "R2"]
        )
        widget.disabled = True
        assert widget.outcome_list.disabled
        assert widget.intervention_list.disabled
        assert widget.reference_time_list.disabled
        assert widget.disabled


class TestModelInterventionAndCohortGroupWidget:
    def test_init(self):
        widget = undertest.ModelInterventionAndCohortGroupWidget(
            cohort_groups={"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]},
            outcome_names=["O1", "O2"],
            intervention_names=["I1", "I2"],
            reference_time_names=["R1", "R2"],
        )

        assert len(widget.children) == 2
        assert widget.cohort == "C1"
        assert widget.cohort_groups == ("C1.1", "C1.2")
        assert widget.outcome == "O1"
        assert widget.intervention == "I1"
        assert widget.reference_time == "R1"

    def test_disable(self):
        widget = undertest.ModelInterventionAndCohortGroupWidget(
            cohort_groups={"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]},
            outcome_names=["O1", "O2"],
            intervention_names=["I1", "I2"],
            reference_time_names=["R1", "R2"],
        )
        widget.disabled = True
        assert widget.model_options.disabled
        assert widget.cohort_list.disabled
        assert widget.disabled


class TestModelScoreComparisonOptionsWidget:
    def test_init(self):
        widget = undertest.ModelScoreComparisonOptionsWidget(
            target_names=["T1", "T2"], score_names=["S1", "S2", "S3", "S4"]
        )
        assert len(widget.children) == 2
        assert "Model Options" in widget.title.value
        assert widget.target == "T1"
        assert widget.scores == (
            "S1",
            "S2",
        )

    def test_disable(self):
        widget = undertest.ModelScoreComparisonOptionsWidget(target_names=["T1", "T2"], score_names=["S1", "S2"])
        widget.disabled = True
        assert widget.target_list.disabled
        assert widget.score_list.disabled
        assert widget.disabled

    def test_disabled_by_list_size(self):
        widget = undertest.ModelScoreComparisonOptionsWidget(target_names=["T1"], score_names=["S1", "S2"])
        widget.disabled = False
        assert widget.target_list.disabled  # only one target
        assert not widget.disabled

    def test_init_per_context(self):
        widget = undertest.ModelScoreComparisonOptionsWidget(
            target_names=["T1", "T2"], score_names=["S1", "S2", "S3", "S4"], per_context=True
        )
        assert len(widget.children) == 3
        assert "Model Options" in widget.title.value
        assert widget.target == "T1"
        assert widget.scores == (
            "S1",
            "S2",
        )
        assert widget.group_scores is True

    def test_disable_per_context(self):
        widget = undertest.ModelScoreComparisonOptionsWidget(
            target_names=["T1", "T2"], score_names=["S1", "S2"], per_context=False
        )
        assert widget.group_scores is False
        widget.disabled = True
        assert widget.target_list.disabled
        assert widget.score_list.disabled
        assert widget.per_context_checkbox.disabled
        assert widget.disabled


class TestModelTargetComparisonOptionsWidget:
    def test_init(self):
        widget = undertest.ModelTargetComparisonOptionsWidget(
            target_names=["T1", "T2", "T3"], score_names=["S1", "S2"]
        )
        assert len(widget.children) == 2
        assert "Model Options" in widget.title.value
        assert widget.targets == (
            "T1",
            "T2",
        )
        assert widget.score == "S1"

    def test_disable(self):
        widget = undertest.ModelTargetComparisonOptionsWidget(target_names=["T1", "T2"], score_names=["S1", "S2"])
        widget.disabled = True
        assert widget.target_list.disabled
        assert widget.score_list.disabled
        assert widget.disabled

    def test_init_per_context(self):
        widget = undertest.ModelTargetComparisonOptionsWidget(
            target_names=["T1", "T2", "T3"], score_names=["S1", "S2"], per_context=True
        )
        assert len(widget.children) == 3
        assert "Model Options" in widget.title.value
        assert widget.targets == (
            "T1",
            "T2",
        )
        assert widget.score == "S1"
        assert widget.group_scores is True

    def test_disable_per_context(self):
        widget = undertest.ModelTargetComparisonOptionsWidget(
            target_names=["T1", "T2"], score_names=["S1", "S2"], per_context=False
        )
        assert widget.group_scores is False
        widget.disabled = True
        assert widget.target_list.disabled
        assert widget.score_list.disabled
        assert widget.per_context_checkbox.disabled
        assert widget.disabled

    def test_disabled_by_list_size(self):
        widget = undertest.ModelTargetComparisonOptionsWidget(target_names=["T1", "T2"], score_names=["S1"])
        widget.disabled = False
        assert widget.score_list.disabled  # only one score
        assert not widget.disabled


class TestModelScoreComparisonAndCohortsWidget:
    def test_init(self):
        widget = undertest.ModelScoreComparisonAndCohortsWidget(
            cohort_groups={"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]},
            target_names=["T1", "T2"],
            score_names=["S1", "S2", "S3", "S4"],
        )
        assert len(widget.children) == 2
        assert widget.cohorts == {}
        assert widget.target == "T1"
        assert widget.scores == (
            "S1",
            "S2",
        )
        assert widget.group_scores is False

    def test_disable(self):
        widget = undertest.ModelScoreComparisonAndCohortsWidget(
            cohort_groups={"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]},
            target_names=["T1", "T2"],
            score_names=["S1", "S2"],
        )
        widget.disabled = True
        assert widget.model_options.disabled
        assert widget.cohort_list.disabled
        assert widget.disabled


class TestModelTargetComparisonAndCohortsWidget:
    def test_init(self):
        widget = undertest.ModelTargetComparisonAndCohortsWidget(
            cohort_groups={"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]},
            target_names=["T1", "T2", "T3"],
            score_names=["S1", "S2"],
        )
        assert len(widget.children) == 2
        assert widget.cohorts == {}
        assert widget.targets == (
            "T1",
            "T2",
        )
        assert widget.score == "S1"
        assert widget.group_scores is False

    def test_disable(self):
        widget = undertest.ModelTargetComparisonAndCohortsWidget(
            cohort_groups={"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]},
            target_names=["T1", "T2"],
            score_names=["S1", "S2"],
        )
        widget.disabled = True
        assert widget.model_options.disabled
        assert widget.cohort_list.disabled
        assert widget.disabled


# endregion
# region Test Exploration Widgets
class TestExplorationSubpopulationWidget:
    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_init(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}
        plot_function = Mock(return_value="some result")
        plot_function.__name__ = "plot_function"
        plot_function.__module__ = "test_explore"

        widget = undertest.ExplorationSubpopulationWidget(title="Subpopulation", plot_function=plot_function)

        assert widget.disabled is False
        assert widget.update_plot_widget.disabled
        assert widget.current_plot_code == "test_explore.plot_function({})"
        plot_function.assert_called_once_with({})  # default value

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_option_update(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}
        plot_function = Mock(return_value="some result")
        plot_function.__name__ = "plot_function"
        plot_function.__module__ = "test_explore"

        widget = undertest.ExplorationSubpopulationWidget(title="Subpopulation", plot_function=plot_function)

        widget.option_widget.value = {
            "C2": [
                "C2.1",
            ]
        }
        widget.update_plot()
        plot_function.assert_called_with(
            {
                "C2": [
                    "C2.1",
                ]
            }
        )  # updated value


class TestExplorationModelSubgroupEvaluationWidget:
    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_init(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}
        fake_seismo.thresholds = [0.1, 0.2]
        fake_seismo.target_cols = ["T1", "T2"]
        fake_seismo.output_list = ["S1", "S2"]

        plot_function = Mock(return_value="some result")
        plot_function.__name__ = "plot_function"
        plot_function.__module__ = "test_explore"

        widget = undertest.ExplorationModelSubgroupEvaluationWidget(
            title="Unit Test Title", plot_function=plot_function
        )

        assert widget.disabled is False
        assert widget.update_plot_widget.disabled
        assert widget.current_plot_code == "test_explore.plot_function({}, 'T1', 'S1', [0.2, 0.1], per_context=False)"
        plot_function.assert_called_once_with({}, "T1", "S1", [0.2, 0.1], per_context=False)  # default value

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_option_update(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}
        fake_seismo.thresholds = [0.1, 0.2]
        fake_seismo.target_cols = ["T1", "T2"]
        fake_seismo.output_list = ["S1", "S2"]

        plot_function = Mock(return_value="some result")
        plot_function.__name__ = "plot_function"
        plot_function.__module__ = "test_explore"

        widget = undertest.ExplorationModelSubgroupEvaluationWidget(
            title="Unit Test Title", plot_function=plot_function
        )

        widget.option_widget.cohort_list.value = {"C2": ("C2.1",)}
        widget.update_plot()
        plot_function.assert_called_with({"C2": ("C2.1",)}, "T1", "S1", [0.2, 0.1], per_context=False)  # updated value


class TestExplorationCohortSubclassEvaluationWidget:
    @pytest.mark.parametrize(
        "threshold_handling,thresholds",
        [("all", [0.2, 0.1]), ("max", 0.2), ("min", 0.1), (None, "")],
    )
    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_init(self, mock_seismo, threshold_handling, thresholds):
        fake_seismo = mock_seismo()
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}
        fake_seismo.thresholds = [0.1, 0.2]
        fake_seismo.target_cols = ["T1", "T2"]
        fake_seismo.output_list = ["S1", "S2"]

        plot_function = Mock(return_value="some result")
        plot_function.__name__ = "plot_function"
        plot_function.__module__ = "test_explore"

        widget = undertest.ExplorationCohortSubclassEvaluationWidget(
            title="Unit Test Title", plot_function=plot_function, threshold_handling=threshold_handling
        )

        assert widget.disabled is False
        assert widget.update_plot_widget.disabled
        expected_code = "test_explore.plot_function('C1', ('C1.1', 'C1.2'), 'T1', 'S1',"
        if thresholds:
            expected_code += f" {thresholds}, per_context=False)"
            plot_function.assert_called_once_with(
                "C1", ("C1.1", "C1.2"), "T1", "S1", thresholds, per_context=False
            )  # default value
        else:
            expected_code += " per_context=False)"
            plot_function.assert_called_once_with(
                "C1", ("C1.1", "C1.2"), "T1", "S1", per_context=False
            )  # default value
        assert widget.current_plot_code == expected_code


class TestExplorationCohortOutcomeInterventionEvaluationWidget:
    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_init(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.predict_time = "pred_time"
        fake_seismo.comparison_time = "comp_time"
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}
        fake_config = Mock(
            outcomes=Mock(keys=Mock(return_value=["O1", "O2"])),
            interventions=Mock(keys=Mock(return_value=["I1", "I2"])),
        )
        fake_seismo.config = fake_config

        plot_function = Mock(return_value="some result")
        plot_function.__name__ = "plot_function"
        plot_function.__module__ = "test_explore"

        widget = undertest.ExplorationCohortOutcomeInterventionEvaluationWidget(
            title="Unit Test Title", plot_function=plot_function
        )

        assert widget.disabled is False
        assert widget.update_plot_widget.disabled
        assert (
            widget.current_plot_code == "test_explore.plot_function('C1', ('C1.1', 'C1.2'), 'O1', 'I1', 'pred_time')"
        )
        plot_function.assert_called_once_with("C1", ("C1.1", "C1.2"), "O1", "I1", "pred_time")  # default value


class TestExplorationScoreComparisonByCohortWidget:
    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_init(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}
        fake_seismo.thresholds = [0.1, 0.2]
        fake_seismo.target_cols = ["T1", "T2"]
        fake_seismo.output_list = ["S1", "S2"]

        plot_function = Mock(return_value="some result")
        plot_function.__name__ = "plot_function"
        plot_function.__module__ = "test_explore"

        widget = undertest.ExplorationScoreComparisonByCohortWidget(
            title="Unit Test Title", plot_function=plot_function
        )

        assert widget.disabled is False
        assert widget.update_plot_widget.disabled
        assert widget.current_plot_code == "test_explore.plot_function({}, 'T1', ('S1', 'S2'), per_context=False)"
        plot_function.assert_called_once_with({}, "T1", ("S1", "S2"), per_context=False)  # default value


class TestExplorationTargetComparisonByCohortWidget:
    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_init(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}
        fake_seismo.thresholds = [0.1, 0.2]
        fake_seismo.target_cols = ["T1", "T2"]
        fake_seismo.output_list = ["S1", "S2"]

        plot_function = Mock(return_value="some result")
        plot_function.__name__ = "plot_function"
        plot_function.__module__ = "test_explore"

        widget = undertest.ExplorationTargetComparisonByCohortWidget(
            title="Unit Test Title", plot_function=plot_function
        )

        assert widget.disabled is False
        assert widget.update_plot_widget.disabled
        assert widget.current_plot_code == "test_explore.plot_function({}, ('T1', 'T2'), 'S1', per_context=False)"
        plot_function.assert_called_once_with({}, ("T1", "T2"), "S1", per_context=False)  # default value


# endregion


# region Test Binary Model Metric Exploration widgets
class FakeMetricGenerator(MetricGenerator):
    def __init__(self):
        def metric_function(dataframe, metric_names, **kwargs):
            return {"Accuracy": 0.8, "F1": 0.7, "Precision": 0.6, "Recall": 0.5}

        super().__init__(["Accuracy", "F1", "Precision", "Recall"], metric_function, ["Precision", "Recall"])


class TestBinaryModelMetricOptions:
    def test_init(self):
        metric_generator = FakeMetricGenerator()
        widget = undertest.BinaryModelMetricOptions(
            metric_generator,
            {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]},
            ["T1", "T2"],
            ["S1", "S2"],
            default_metrics=["Precision", "Recall"],
        )
        assert widget.disabled is False
        assert widget.metrics == ("Precision", "Recall")
        assert widget.metric_list.options == tuple(metric_generator.metric_names)
        assert widget.target == "T1"
        assert widget.score == "S1"

    def test_disable(self):
        metric_generator = FakeMetricGenerator()
        widget = undertest.BinaryModelMetricOptions(
            metric_generator,
            {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]},
            ["T1", "T2"],
            ["S1", "S2"],
            default_metrics=["Precision", "Recall"],
        )
        widget.disabled = True
        assert widget.metric_list.disabled
        assert widget.model_options.disabled
        assert widget.cohort_list.disabled
        assert widget.disabled


class TestExplorationMetricWidget:
    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_init(self, mock_seismo):
        metric_generator = FakeMetricGenerator()
        fake_seismo = mock_seismo()
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}
        fake_seismo.thresholds = [0.1, 0.2]
        fake_seismo.target_cols = ["T1", "T2"]
        fake_seismo.output_list = ["S1", "S2"]

        plot_function = Mock(return_value="some result")
        plot_function.__name__ = "plot_function"
        plot_function.__module__ = "test_explore"

        widget = undertest.ExplorationMetricWidget(
            title="Unit Test Title", metric_generator=metric_generator, plot_function=plot_function
        )

        assert widget.disabled is False
        assert widget.update_plot_widget.disabled
        assert (
            widget.current_plot_code
            == "test_explore.plot_function(MetricGenerator("
            + "metric_names=['Accuracy', 'F1', 'Precision', 'Recall'], metric_fn=metric_function), "
            + "('Precision', 'Recall'), {}, 'T1', 'S1', per_context=False)"
        )
        plot_function.assert_called_once_with(
            metric_generator, ("Precision", "Recall"), {}, "T1", "S1", per_context=False
        )

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_option_update(self, mock_seismo):
        metric_generator = FakeMetricGenerator()
        fake_seismo = mock_seismo()
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}
        fake_seismo.thresholds = [0.1, 0.2]
        fake_seismo.target_cols = ["T1", "T2"]
        fake_seismo.output_list = ["S1", "S2"]

        plot_function = Mock(return_value="some result")
        plot_function.__name__ = "plot_function"
        plot_function.__module__ = "test_explore"

        widget = undertest.ExplorationMetricWidget(
            title="Unit Test Title", metric_generator=metric_generator, plot_function=plot_function
        )

        widget.option_widget.metric_list.value = ("F1",)
        widget.option_widget.model_options.target_list.value = "T2"
        widget.option_widget.model_options.score_list.value = "S2"
        widget.update_plot()

        assert (
            widget.current_plot_code
            == "test_explore.plot_function(MetricGenerator("
            + "metric_names=['Accuracy', 'F1', 'Precision', 'Recall'], metric_fn=metric_function), "
            + "('F1',), {}, 'T2', 'S2', per_context=False)"
        )
        plot_function.assert_called_with(metric_generator, ("F1",), {}, "T2", "S2", per_context=False)


class TestExploreBinaryModelAnalytics:
    @patch("seismometer.table.analytics_table.binary_analytics_table", return_value="some result")
    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_init(self, mock_seismo, mock_plot_function):
        fake_seismo = mock_seismo()
        fake_seismo.get_binary_targets.return_value = ["T1_Value", "T2_Value"]
        fake_seismo.output_list = ["S1", "S2"]
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}

        mock_plot_function.__name__ = "plot_function"
        mock_plot_function.__module__ = "test_explore"

        widget = ExploreBinaryModelAnalytics(title="Unit Test Title")

        assert widget.disabled is False
        assert widget.current_plot_code == "No plot generated."
        widget.update_plot()
        assert (
            widget.current_plot_code
            == "test_explore.plot_function(('T1_Value', 'T2_Value'), ('S1', 'S2'), 'Threshold', (0.8, 0.2), "
            + "['Positives', 'Prevalence', 'AUROC', 'AUPRC', 'Accuracy', 'PPV', 'Sensitivity', 'Specificity', "
            + "'Flag Rate', 'Threshold'], 'Score', {}, "
            + "title='Unit Test Title', per_context=False)"
        )
        mock_plot_function.assert_called_once_with(
            ("T1_Value", "T2_Value"),
            ("S1", "S2"),
            "Threshold",
            (0.8, 0.2),
            [
                "Positives",
                "Prevalence",
                "AUROC",
                "AUPRC",
                "Accuracy",
                "PPV",
                "Sensitivity",
                "Specificity",
                "Flag Rate",
                "Threshold",
            ],
            "Score",
            {},
            title="Unit Test Title",
            per_context=False,
        )

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_generate_plot_args(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.get_binary_targets.return_value = ["T1_Value", "T2_Value"]
        fake_seismo.output_list = ["S1", "S2"]
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}

        widget = ExploreBinaryModelAnalytics(title="Unit Test Title")

        args, kwargs = widget.generate_plot_args()
        assert args == (
            ("T1_Value", "T2_Value"),
            ("S1", "S2"),
            "Threshold",
            (0.8, 0.2),
            [
                "Positives",
                "Prevalence",
                "AUROC",
                "AUPRC",
                "Accuracy",
                "PPV",
                "Sensitivity",
                "Specificity",
                "Flag Rate",
                "Threshold",
            ],
            "Score",
            {},
        )
        assert kwargs == {"title": "Unit Test Title", "per_context": False}


class TestAnalyticsTableOptionsWidget:
    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_init(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.get_binary_targets.return_value = ["T1", "T2"]
        fake_seismo.output_list = ["S1", "S2"]
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}

        widget = AnalyticsTableOptionsWidget(
            target_cols=("T1", "T2"),
            score_cols=("S1", "S2"),
            metric="Threshold",
            metric_values=[0.8, 0.2],
            metrics_to_display=("Accuracy", "PPV"),
            cohort_dict=fake_seismo.available_cohort_groups,
            title="Unit Test Title",
        )

        assert widget._target_cols.value == ("T1", "T2")
        assert widget._score_cols.value == ("S1", "S2")
        assert widget._metric.value == "Threshold"
        assert widget._metric_values.value == {"Metric Value 1": 0.8, "Metric Value 2": 0.2}
        assert widget._metrics_to_display.value == ("Accuracy", "PPV")
        assert widget._group_by.value == "Score"
        assert widget.per_context_checkbox.value is False
        assert widget._cohort_dict.value == {}

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_disabled_property(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.get_binary_targets.return_value = ["T1", "T2"]
        fake_seismo.output_list = ["S1", "S2"]
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}

        widget = AnalyticsTableOptionsWidget(
            target_cols=("T1", "T2"),
            score_cols=("S1", "S2"),
            metric="Threshold",
            cohort_dict=fake_seismo.available_cohort_groups,
        )
        widget.disabled = True
        assert widget._target_cols.disabled is True
        assert widget._score_cols.disabled is True
        assert widget._metric.disabled is True
        assert widget._metric_values.disabled is True
        assert widget._metrics_to_display.disabled is True
        assert widget._group_by.disabled is True
        assert widget.per_context_checkbox.disabled is True
        assert widget._cohort_dict.disabled is True

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_on_value_changed(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.get_binary_targets.return_value = ["T1", "T2"]
        fake_seismo.output_list = ["S1", "S2"]
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}

        widget = AnalyticsTableOptionsWidget(
            target_cols=("T1", "T2"),
            score_cols=("S1", "S2"),
            metric="Threshold",
            cohort_dict=fake_seismo.available_cohort_groups,
        )
        widget._target_cols.value = ("T1",)
        widget._score_cols.value = ("S1",)
        widget._metric.value = "Sensitivity"
        widget._metric_values.value = {"Metric Value 1": 0.9, "Metric Value 2": 0.1}
        widget._metrics_to_display.value = ("Accuracy",)
        widget._group_by.value = "Target"
        widget._cohort_dict.value = {
            "C1": [
                "C1.1",
            ]
        }
        widget.per_context_checkbox.value = True

        expected_value = {
            "target_cols": ("T1",),
            "score_cols": ("S1",),
            "metric": "Sensitivity",
            "metric_values": {"Metric Value 1": 0.9, "Metric Value 2": 0.1},
            "metrics_to_display": ("Accuracy",),
            "group_by": "Target",
            "cohort_dict": {
                "C1": [
                    "C1.1",
                ]
            },
            "group_scores": True,
        }
        assert widget.value == expected_value

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_model_options_widget(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.get_binary_targets.return_value = ["T1", "T2"]
        fake_seismo.output_list = ["S1", "S2"]
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}

        model_options_widget = ipywidgets.Dropdown(
            options=["Val1", "Val2"],
            value="Val1",
            description="Test model options",
        )
        widget = AnalyticsTableOptionsWidget(
            target_cols=("T1", "T2"),
            score_cols=("S1", "S2"),
            metric="Threshold",
            cohort_dict=fake_seismo.available_cohort_groups,
            model_options_widget=model_options_widget,
        )
        assert widget.model_options_widget == model_options_widget

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_group_scores(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.get_binary_targets.return_value = ["T1", "T2"]
        fake_seismo.output_list = ["S1", "S2"]
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}

        widget = AnalyticsTableOptionsWidget(
            target_cols=("T1", "T2"),
            score_cols=("S1", "S2"),
            metric="Threshold",
            cohort_dict=fake_seismo.available_cohort_groups,
        )
        widget.per_context_checkbox.value = True
        assert widget.group_scores is True

        widget.per_context_checkbox.value = False
        assert widget.group_scores is False

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_edge_cases(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.get_binary_targets.return_value = []
        fake_seismo.output_list = []
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}

        widget = AnalyticsTableOptionsWidget(
            target_cols=(),
            score_cols=(),
            metric="Threshold",
            metric_values=[],
            metrics_to_display=(),
            cohort_dict={},
            title="Unit Test Title",
        )

        assert widget._target_cols.value == ()
        assert widget._score_cols.value == ()
        assert widget._metric.value == "Threshold"
        assert widget._metric_values.value == {"Metric Value 1": 0.8, "Metric Value 2": 0.2}
        assert widget._metrics_to_display.value == (
            "Positives",
            "Prevalence",
            "AUROC",
            "AUPRC",
            "Accuracy",
            "PPV",
            "Sensitivity",
            "Specificity",
            "Flag Rate",
            "Threshold",
        )
        assert widget._group_by.value == "Score"
        assert widget.per_context_checkbox.value is False

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_state_changes(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.get_binary_targets.return_value = ["T1", "T2"]
        fake_seismo.output_list = ["S1", "S2"]
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}

        widget = AnalyticsTableOptionsWidget(
            target_cols=("T1", "T2"),
            score_cols=("S1", "S2"),
            metric="Threshold",
            metric_values=[0.8, 0.2],
            metrics_to_display=("Accuracy", "PPV"),
            cohort_dict=fake_seismo.available_cohort_groups,
            title="Unit Test Title",
        )

        # Change the state of the widget
        widget._target_cols.value = ("T2",)
        widget._score_cols.value = ("S2",)
        widget._metric.value = "Sensitivity"
        widget._metric_values.value = {"Metric Value 1": 0.9, "Metric Value 2": 0.1}
        widget._metrics_to_display.value = ("PPV",)
        widget._group_by.value = "Target"
        widget._cohort_dict.value = {
            "C1": [
                "C1.2",
            ]
        }
        widget.per_context_checkbox.value = True

        # Verify the state changes
        assert widget._target_cols.value == ("T2",)
        assert widget._score_cols.value == ("S2",)
        assert widget._metric.value == "Sensitivity"
        assert widget._metric_values.value == {"Metric Value 1": 0.9, "Metric Value 2": 0.1}
        assert widget._metrics_to_display.value == ("PPV",)
        assert widget._group_by.value == "Target"
        assert widget._cohort_dict.value == {
            "C1": [
                "C1.2",
            ]
        }
        assert widget.per_context_checkbox.value is True


class TestExploreCategoricalPlots:
    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_init(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.metric_groups = {"Group1": ["Metric1", "Metric2"]}
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}
        fake_seismo.get_ordinal_categorical_groups = lambda x: ["Group1"]
        fake_seismo.get_ordinal_categorical_metrics = lambda x: ["Metric1", "Metric2"]

        widget = ExploreCategoricalPlots(title="Unit Test Title")

        assert widget.disabled is False

        expected_plot_code = (
            "seismometer.controls.categorical.ordinal_categorical_plot(['Metric1', 'Metric2'], "
            + "{'C1': ['C1.1', 'C1.2'], 'C2': ['C2.1', 'C2.2']}, title='Unit Test Title')"
        )
        assert widget.current_plot_code == expected_plot_code
        widget.update_plot()
        assert widget.current_plot_code == expected_plot_code

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_generate_plot_args(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.metric_groups = {"Group1": ["Metric1", "Metric2"]}
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}
        fake_seismo.get_ordinal_categorical_groups = lambda x: ["Group1"]
        fake_seismo.get_ordinal_categorical_metrics = lambda x: ["Metric1", "Metric2"]

        widget = ExploreCategoricalPlots(title="Unit Test Title")

        args, kwargs = widget.generate_plot_args()
        assert args == (["Metric1", "Metric2"], {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]})
        assert kwargs == {"title": "Unit Test Title"}


class TestCategoricalOptionsWidget:
    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_init(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.metric_groups = {"Group1": ["Metric1", "Metric2"]}
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}
        fake_seismo.get_ordinal_categorical_groups = lambda x: ["Group1"]
        fake_seismo.get_ordinal_categorical_metrics = lambda x: ["Metric1", "Metric2"]

        widget = CategoricalOptionsWidget(
            metric_groups=["Group1"], cohort_dict=fake_seismo.available_cohort_groups, title="Unit Test Title"
        )

        assert widget._metric_groups.value == ("Group1",)
        assert widget._metrics.value == ("Metric1", "Metric2")
        assert widget._cohort_dict.value == {}
        assert widget.all_cohorts == fake_seismo.available_cohort_groups

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_disabled_property(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.metric_groups = {"Group1": ["Metric1", "Metric2"]}
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}
        fake_seismo.get_ordinal_categorical_groups = lambda x: ["Group1"]
        fake_seismo.get_ordinal_categorical_metrics = lambda x: ["Metric1", "Metric2"]

        widget = CategoricalOptionsWidget(metric_groups=["Group1"], cohort_dict=fake_seismo.available_cohort_groups)
        widget.disabled = True
        assert widget._metric_groups.disabled is True
        assert widget._metrics.disabled is True
        assert widget._cohort_dict.disabled is True

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_on_value_changed(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.metric_groups = {"Group1": ("Metric1", "Metric2")}
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}
        fake_seismo.get_ordinal_categorical_groups = lambda x: ["Group1"]
        fake_seismo.get_ordinal_categorical_metrics = lambda x: ["Metric1", "Metric2"]

        widget = CategoricalOptionsWidget(metric_groups=["Group1"], cohort_dict=fake_seismo.available_cohort_groups)
        widget._metric_groups.value = ["Group1"]
        widget._metrics.value = ["Metric1"]
        widget._cohort_dict.value = {"C1": ["C1.1"]}

        expected_value = {"metric_groups": ("Group1",), "metrics": ("Metric1",), "cohort_dict": {"C1": ["C1.1"]}}
        assert widget.value == expected_value

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_model_options_widget(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.metric_groups = {"Group1": ("Metric1", "Metric2")}
        fake_seismo.available_cohort_groups = {"C1": ["C1.1", "C1.2"], "C2": ["C2.1", "C2.2"]}
        fake_seismo.get_ordinal_categorical_groups = lambda x: ["Group1"]
        fake_seismo.get_ordinal_categorical_metrics = lambda x: ["Metric1", "Metric2"]

        model_options_widget = ipywidgets.Dropdown(
            options=["Val1", "Val2"],
            value="Val1",
            description="Test model options",
        )
        widget = CategoricalOptionsWidget(
            metric_groups=["Group1"],
            cohort_dict=fake_seismo.available_cohort_groups,
            model_options_widget=model_options_widget,
        )
        assert widget.model_options_widget == model_options_widget


class TestExploreSingleCategoricalPlots:
    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_init(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.metric_groups = {"Group1": ["Metric1", "Metric2"]}
        fake_seismo.available_cohort_groups = {"Age": ["20-30", "30-40"]}
        fake_seismo.get_ordinal_categorical_groups = lambda x: ["Group1"]
        fake_seismo.get_ordinal_categorical_metrics = lambda x: ["Metric1", "Metric2"]

        # Mock the initialization parameters
        fake_seismo.return_value.config = Mock()
        fake_seismo.return_value.dataloader = Mock()

        widget = ExploreSingleCategoricalPlots(title="Unit Test Title")

        assert widget.disabled is False

        expected_plot_code = (
            "seismometer.controls.categorical_single_column.ordinal_categorical_single_col_plot"
            + "('Metric1', {'Age': ('20-30', '30-40')}, title='Unit Test Title')"
        )
        assert widget.current_plot_code == expected_plot_code
        widget.update_plot()
        assert widget.current_plot_code == expected_plot_code

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_generate_plot_args(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.metric_groups = {"Group1": ["Metric1", "Metric2"]}
        fake_seismo.available_cohort_groups = {"Age": ["20-30", "30-40"]}
        fake_seismo.get_ordinal_categorical_groups = lambda x: ["Group1"]
        fake_seismo.get_ordinal_categorical_metrics = lambda x: ["Metric1", "Metric2"]

        # Mock the initialization parameters
        fake_seismo.return_value.config = Mock()
        fake_seismo.return_value.dataloader = Mock()

        widget = ExploreSingleCategoricalPlots(title="Unit Test Title")

        args, kwargs = widget.generate_plot_args()
        assert args == ("Metric1", {"Age": ("20-30", "30-40")})
        assert kwargs == {"title": "Unit Test Title"}


class TestCategoricalFeedbackSingleColumnOptionsWidget:
    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_init(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.metrics = {"Metric1": "Description1", "Metric2": "Description2"}
        fake_seismo.available_cohort_groups = {"Age": ["20-30", "30-40"]}

        # Mock the initialization parameters
        fake_seismo.return_value.config = Mock()
        fake_seismo.return_value.dataloader = Mock()

        widget = CategoricalFeedbackSingleColumnOptionsWidget(
            metrics=["Metric1", "Metric2"], cohort_groups=fake_seismo.available_cohort_groups, title="Unit Test Title"
        )

        assert widget._metric_col.value == "Metric1"
        assert widget._cohort_list.value == ("Age", ("20-30", "30-40"))
        assert widget.title == "Unit Test Title"

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_disabled_property(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.metrics = {"Metric1": "Description1", "Metric2": "Description2"}
        fake_seismo.available_cohort_groups = {"Age": ["20-30", "30-40"]}

        # Mock the initialization parameters
        fake_seismo.return_value.config = Mock()
        fake_seismo.return_value.dataloader = Mock()

        widget = CategoricalFeedbackSingleColumnOptionsWidget(
            metrics=["Metric1", "Metric2"], cohort_groups=fake_seismo.available_cohort_groups
        )
        widget.disabled = True
        assert widget._metric_col.disabled is True
        assert widget._cohort_list.disabled is True

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_on_value_changed(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.metrics = {"Metric1": "Description1", "Metric2": "Description2"}
        fake_seismo.available_cohort_groups = {"Age": ["20-30", "30-40"]}

        # Mock the initialization parameters
        fake_seismo.return_value.config = Mock()
        fake_seismo.return_value.dataloader = Mock()

        widget = CategoricalFeedbackSingleColumnOptionsWidget(
            metrics=["Metric1", "Metric2"], cohort_groups=fake_seismo.available_cohort_groups
        )
        widget._metric_col.value = "Metric2"
        widget._cohort_list.value = ("Age", ("20-30",))

        expected_value = {"metric_col": "Metric2", "cohort_list": ("Age", ("20-30",))}
        assert widget.value == expected_value

    @patch.object(seismogram, "Seismogram", return_value=Mock())
    def test_model_options_widget(self, mock_seismo):
        fake_seismo = mock_seismo()
        fake_seismo.metrics = {"Metric1": "Description1", "Metric2": "Description2"}
        fake_seismo.available_cohort_groups = {"Age": ["20-30", "30-40"]}

        # Mock the initialization parameters
        fake_seismo.return_value.config = Mock()
        fake_seismo.return_value.dataloader = Mock()

        model_options_widget = ipywidgets.Dropdown(
            options=["Val1", "Val2"],
            value="Val1",
            description="Test model options",
        )
        widget = CategoricalFeedbackSingleColumnOptionsWidget(
            metrics=["Metric1", "Metric2"],
            cohort_groups=fake_seismo.available_cohort_groups,
            model_options_widget=model_options_widget,
        )
        assert widget.model_options_widget == model_options_widget


# endregion
