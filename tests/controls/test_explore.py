import ipywidgets
import pytest

import seismometer.controls.explore as undertest


class TestUpdatePlotWidget:
    def test_init(self):
        widget = undertest.UpdatePlotWidget()
        assert widget.plot_button.description == undertest.UpdatePlotWidget.UPDATE_PLOTS
        assert widget.plot_button.disabled is False
        assert widget.code_checkbox.description == "show code"
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

        with pytest.raises(NotImplementedError):
            undertest.ExplorationWidget("ExploreTest", option_widget, lambda x: x)


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


class TestModelFairnessAuditOptions:
    def test_init(self):
        widget = undertest.ModelFairnessAuditOptions(
            target_names=["T1", "T2"],
            score_names=["S1", "S2"],
            score_threshold=0.1,
            per_context=True,
            fairness_metrics=None,
            fairness_threshold=1.25,
        )

        assert widget.target == "T1"
        assert widget.score == "S1"
        assert widget.score_threshold == 0.1
        assert widget.group_scores is True
        assert widget.metrics == ("pprev", "tpr", "fpr")
        assert widget.fairness_threshold == 1.25

    def test_disable(self):
        widget = undertest.ModelFairnessAuditOptions(
            target_names=["T1", "T2"],
            score_names=["S1", "S2"],
            score_threshold=0.1,
            per_context=True,
            fairness_metrics=None,
            fairness_threshold=1.25,
        )
        widget.disabled = True
        assert widget.model_options.disabled
        assert widget.fairness_slider.disabled
        assert widget.fairness_list.disabled
        assert widget.disabled
