import logging

import pytest
from conftest import res  # noqa: F401
from IPython.display import HTML, SVG

import seismometer.html.template as undertest


def read_template_from_file(filename):
    html_str = "".join(open(filename).readlines())
    return HTML(html_str)


@pytest.fixture
def info_template_no_plot_help(res):
    return read_template_from_file(res / "rendered_templates/info_no_plot_help.html")


@pytest.fixture
def info_template(res):
    return read_template_from_file(res / "rendered_templates/info.html")


@pytest.fixture
def cohort_summaries_template(res):
    return read_template_from_file(res / "rendered_templates/cohort_summaries_template.html")


class Test_Templates:
    def test_nonexistent_template(self, caplog):
        with caplog.at_level(logging.WARNING, logger="seismometer"):
            undertest.render_into_template("unknown")
        assert "HTML template unknown not found" in caplog.text

    def test_info_template_no_plot_help(self, info_template_no_plot_help):
        info_vals = {
            "tables": [
                {
                    "name": "sg.dataframe",
                    "description": "Scores, features, configured demographics, and merged events for each prediction",
                    "num_rows": 1,
                    "num_cols": 1,
                }
            ],
            "num_predictions": "num_predictions",
            "num_entities": "num_entities",
            "start_date": "start_date",
            "end_date": "end_date",
            "plot_help": False,
        }

        assert undertest.render_info_template(info_vals).data == info_template_no_plot_help.data

    def test_info_template(self, info_template):
        info_vals = {
            "tables": [
                {
                    "name": "sg.dataframe",
                    "description": "Scores, features, configured demographics, and merged events for each prediction",
                    "num_rows": 1,
                    "num_cols": 1,
                }
            ],
            "num_predictions": "num_predictions",
            "num_entities": "num_entities",
            "start_date": "start_date",
            "end_date": "end_date",
            "plot_help": True,
        }

        assert undertest.render_info_template(info_vals).data == info_template.data

    def test_cohort_summaries_template(self, cohort_summaries_template):
        assert (
            undertest.render_cohort_summary_template(
                {"cohort1": ["dataframe1!", "dataframe2!"], "cohort2": ["dataframe1!", "dataframe2!"]}
            ).data
            == cohort_summaries_template.data
        )

    def test_title_message_template(self):
        html_source = undertest.render_title_message("A Title", "The Message").data
        assert "A Title" in html_source
        assert "The Message" in html_source

    def test_censored_plot_template(self):
        html_source = undertest.render_censored_plot_message(3).data
        assert "Censored" in html_source
        assert "There are 3 or fewer observations." in html_source

    def test_censored_data_template(self):
        html_source = undertest.render_censored_data_message(Exception("Somthing Bad Happened")).data
        assert "Censored" in html_source
        assert "Somthing Bad Happened" in html_source

    def test_title_image_template(self):
        svg_data = """
            <svg xmlns="http://www.w3.org/2000/svg" width="100" height="50" viewBox="0 0 100 50">
                <text x="10" y="30" font-family="Arial" font-size="20" fill="black">
                    svg string
                </text>
            </svg>"""
        html_source = undertest.render_title_with_image("A Title", SVG(svg_data)).data
        assert "A Title" in html_source
        assert "svg string" in html_source
