import logging
from functools import partial
from typing import Optional

from IPython.display import display
from ipywidgets import Button, HBox, Output, VBox

from seismometer.controls.selection import MultiSelectionListWidget
from seismometer.data.filter import filter_rule_from_cohort_dictionary

logger = logging.getLogger("seismometer")

GENERATE_REPORT = "Generate Report"
GENERATING_REPORT = "Generating Report..."


class ComparisonReportGenerator:
    def __init__(self, selections: dict[str, tuple[any]], exclude_cols: Optional[list[str]] = None):
        self.selectors: list[MultiSelectionListWidget] = []
        self.exclude_cols = exclude_cols or []

        for side in ["Left", "Right"]:
            options = selections
            widget = MultiSelectionListWidget(options=options, title=f"Select {side} Cohort")
            self.selectors.append(widget)

        self.output = Output()
        self.button = Button(description=GENERATE_REPORT, button_style="primary")
        self.button.on_click(partial(self._generate_comparison_report, self))

    def show(self):
        display(VBox(children=[HBox(children=self.selectors), self.button, self.output]))

    def nth_cohort(self, n: int):
        return self.selectors[n].value

    def nth_selection_text(self, n: int):
        return self.selectors[n].get_selection_text()

    def _generate_comparison_report(self, *args):
        self.output.clear_output()
        with self.output:
            self.button.description = GENERATING_REPORT
            self.button.disabled = True
            from seismometer.report.profiling import ComparisonReportWrapper
            from seismometer.seismogram import Seismogram

            sg = Seismogram()
            exclude_cols = self.exclude_cols + sg.entity_keys

            l_title = self.nth_selection_text(0)
            l_groups = self.nth_cohort(0)
            l_cohort = filter_rule_from_cohort_dictionary(l_groups)

            r_title = self.nth_selection_text(1)
            r_groups = self.nth_cohort(1)
            r_cohort = filter_rule_from_cohort_dictionary(r_groups)

            if l_cohort is None or r_cohort is None:
                logger.warning(
                    "No comparison report generated. Select at least one cohort for the left and the right."
                )
                self.button.description = GENERATE_REPORT
                self.button.disabled = False
                return

            l_df = l_cohort.filter(sg.dataframe)
            r_df = r_cohort.filter(sg.dataframe)

            if l_df.empty:
                logger.warning(
                    f"No comparsion report generated. The left selection ({l_title}) has no data to profile."
                )
                self.button.description = GENERATE_REPORT
                self.button.disabled = False
                return

            if r_df.empty:
                logger.warning(
                    f"No comparsion report generated. The right selection ({r_title}) has no data to profile."
                )
                self.button.description = GENERATE_REPORT
                self.button.disabled = False
                return

            wrapper = ComparisonReportWrapper(
                l_df=l_df,
                r_df=r_df,
                output_path=sg.output_path,
                l_title=l_title,
                r_title=r_title,
                exclude_cols=exclude_cols,
                base_title="Feature Report",
            )

            wrapper.display_report(inline=False)

        self.button.description = GENERATE_REPORT
        self.button.disabled = False
