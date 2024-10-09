from unittest.mock import MagicMock, Mock, patch

import ipywidgets
import pytest

import seismometer.controls.explore as undertest
from seismometer import seismogram


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
            ("seismometer._api", "sm.plot_something(False)"),
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


# endregion
