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

        # This doesnt throw an exception because the exception goes to the output widget
        widget = undertest.ExlorationWidget("ExploreTest", option_widget)

        # This captures the log of the exception
        assert len(caplog.records) == 1
        assert "Subclasses must implement this method" in caplog.records[0].exc_text

        # Test the rest of init
        assert widget.show_code is False
        assert "ExploreTest" in widget.children[0].value
        assert widget.children[1] == option_widget
        assert not widget.disabled

        # Test show code, this will raise an exception
        with pytest.raises(NotImplementedError):
            widget.update_plot_widget.code_checkbox.value = True

        # Test plot button doesnt disable for the base widget
        option_widget.value = True
        assert not widget.disabled


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
        assert not widget.group_scores

    def test_no_combine_scores_checkbox(self):
        widget = undertest.ModelOptionsWidget(
            target_names=["T1", "T2"], score_names=["S1", "S2"], thresholds={"T1": 0.1, "T2": 0.2}, per_context=False
        )
        assert len(widget.children) == 4
        assert "Model Options" in widget.title.value
        assert widget.target == "T1"
        assert widget.score == "S1"
        assert widget.thresholds == (0.1, 0.2)
        assert widget.per_context_checkbox is None

    def test_no_score_thresholds(self):
        widget = undertest.ModelOptionsWidget(target_names=["T1", "T2"], score_names=["S1", "S2"], per_context=False)
        assert len(widget.children) == 3
        assert "Model Options" in widget.title.value
        assert widget.target == "T1"
        assert widget.score == "S1"
        assert widget.threshold_list is None
        assert widget.per_context_checkbox is None

    def test_per_context_checkbox(self):
        widget = undertest.ModelOptionsWidget(target_names=["T1", "T2"], score_names=["S1", "S2"], per_context=True)

        widget.per_context_checkbox.value = True
        assert widget.group_scores
