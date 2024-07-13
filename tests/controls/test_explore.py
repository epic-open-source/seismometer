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
