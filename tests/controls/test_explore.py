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
        widget.trigger()
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


class TestExporationBaseClass:
    def test_base_class(self, caplog):
        option_widget = ipywidgets.Checkbox(description="ClickMe")

        with pytest.raises(NotImplementedError):
            undertest.ExlorationWidget("ExploreTest", option_widget)


class TestModelOptionsWidget:
    def test_init_with_all_options(self):
        widget = undertest.ModelOptionsWidget(
            target_names=["T1", "T2"], score_names=["S1", "S2"], thresholds={"T1": 0.1, "T2": 0.2}, per_context=True
        )
        assert len(widget.children) == 5
        assert "Model Options" in widget.title.value
        assert widget.target == "T1"
        assert widget.score == "S1"
        assert widget.thresholds == (0.1, 0.2)
        assert widget.group_scores is True

    def test_init_with_all_options_grouping_off(self):
        widget = undertest.ModelOptionsWidget(
            target_names=["T1", "T2"], score_names=["S1", "S2"], thresholds={"T1": 0.1, "T2": 0.2}, per_context=False
        )
        assert len(widget.children) == 5
        assert "Model Options" in widget.title.value
        assert widget.target == "T1"
        assert widget.score == "S1"
        assert widget.thresholds == (0.1, 0.2)
        assert widget.group_scores is False

    def test_no_combine_scores_checkbox(self):
        widget = undertest.ModelOptionsWidget(
            target_names=["T1", "T2"], score_names=["S1", "S2"], thresholds={"T1": 0.1, "T2": 0.2}
        )
        assert len(widget.children) == 4
        assert "Model Options" in widget.title.value
        assert widget.target == "T1"
        assert widget.score == "S1"
        assert widget.thresholds == (0.1, 0.2)
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
