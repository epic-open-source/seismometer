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


# ============================================================================
# ADDITIONAL EDGE CASE TESTS
# ============================================================================


class TestRenderingEdgeCases:
    """Test edge cases for template rendering functions."""

    def test_render_title_message_with_very_long_strings(self):
        """Test rendering with very long title and message strings."""
        long_title = "A" * 1000
        long_message = "B" * 10000

        html_source = undertest.render_title_message(long_title, long_message).data

        assert long_title in html_source
        assert long_message in html_source
        assert isinstance(html_source, str)

    @pytest.mark.parametrize(
        "special_chars",
        [
            "<script>alert('xss')</script>",
            "&lt;&gt;&amp;&quot;&#39;",
            "Special chars: < > & \" '",
            "Unicode: \u2665 \u2764 \u263A",
            "Newlines:\n\nMultiple\n\nLines",
            "Tabs:\t\tMultiple\t\tTabs",
        ],
    )
    def test_render_title_message_with_special_html_characters(self, special_chars):
        """Test rendering with special HTML characters and entities."""
        html_source = undertest.render_title_message("Title", special_chars).data

        assert "Title" in html_source
        # The message should be present (possibly HTML-escaped by Jinja2)
        assert isinstance(html_source, str)
        assert len(html_source) > 0

    def test_render_title_message_with_empty_strings(self):
        """Test rendering with empty title and message."""
        html_source = undertest.render_title_message("", "").data

        assert isinstance(html_source, str)
        assert len(html_source) > 0  # Should still have HTML structure

    def test_render_censored_plot_message_with_zero_threshold(self):
        """Test render_censored_plot_message with zero threshold."""
        html_source = undertest.render_censored_plot_message(0).data

        assert "Censored" in html_source
        assert "0 or fewer observations" in html_source

    def test_render_censored_plot_message_with_large_threshold(self):
        """Test render_censored_plot_message with large threshold."""
        html_source = undertest.render_censored_plot_message(999999).data

        assert "Censored" in html_source
        assert "999999 or fewer observations" in html_source

    def test_render_censored_data_message_with_empty_string(self):
        """Test render_censored_data_message with empty message."""
        html_source = undertest.render_censored_data_message("").data

        assert "Censored" in html_source
        assert isinstance(html_source, str)

    def test_render_censored_data_message_with_long_message(self):
        """Test render_censored_data_message with very long message."""
        long_message = "X" * 10000
        html_source = undertest.render_censored_data_message(long_message).data

        assert "Censored" in html_source
        assert long_message in html_source

    def test_render_censored_data_message_with_html_in_message(self):
        """Test render_censored_data_message with HTML-like content in message."""
        html_message = "<b>Bold Error</b> <i>Italic Warning</i>"
        html_source = undertest.render_censored_data_message(html_message).data

        assert "Censored" in html_source
        assert isinstance(html_source, str)


class TestRenderTitleWithImageEdgeCases:
    """Test edge cases for render_title_with_image function."""

    def test_render_title_with_empty_svg_data(self):
        """Test render_title_with_image with SVG that has minimal content."""
        # Empty string is not valid XML, so use minimal valid SVG
        minimal_svg = SVG('<svg xmlns="http://www.w3.org/2000/svg"/>')
        html_source = undertest.render_title_with_image("Title", minimal_svg).data

        assert "Title" in html_source
        assert isinstance(html_source, str)

    def test_render_title_with_minimal_svg(self):
        """Test render_title_with_image with minimal valid SVG."""
        minimal_svg = SVG('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
        html_source = undertest.render_title_with_image("Minimal", minimal_svg).data

        assert "Minimal" in html_source
        assert "svg" in html_source

    def test_render_title_with_complex_svg(self):
        """Test render_title_with_image with complex SVG containing multiple elements."""
        complex_svg_data = """
        <svg xmlns="http://www.w3.org/2000/svg" width="200" height="200">
            <circle cx="100" cy="100" r="50" fill="red"/>
            <rect x="50" y="50" width="100" height="100" fill="blue" opacity="0.5"/>
            <text x="100" y="100" text-anchor="middle" fill="white">Complex</text>
            <path d="M 10 10 L 190 190" stroke="green" stroke-width="2"/>
        </svg>"""
        html_source = undertest.render_title_with_image("Complex SVG", SVG(complex_svg_data)).data

        assert "Complex SVG" in html_source
        assert "circle" in html_source or "rect" in html_source
        assert isinstance(html_source, str)

    def test_render_title_with_very_long_title(self):
        """Test render_title_with_image with very long title."""
        long_title = "A" * 1000
        simple_svg = SVG('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
        html_source = undertest.render_title_with_image(long_title, simple_svg).data

        assert long_title in html_source


class TestRenderIntoTemplateEdgeCases:
    """Test edge cases for render_into_template function."""

    def test_render_into_template_with_none_values(self):
        """Test render_into_template with None values."""
        html = undertest.render_into_template("title_message", None)

        assert isinstance(html, HTML)
        assert isinstance(html.data, str)

    def test_render_into_template_with_empty_dict(self):
        """Test render_into_template with empty dictionary."""
        html = undertest.render_into_template("title_message", {})

        assert isinstance(html, HTML)
        assert isinstance(html.data, str)

    def test_render_into_template_with_custom_display_style(self):
        """Test render_into_template with custom display_style."""
        html = undertest.render_into_template(
            "title_message", {"title": "Test", "message": "Message"}, display_style="width: 50%;"
        )

        assert isinstance(html, HTML)
        assert "Test" in html.data


class TestCohortSummaryTemplate:
    """Test cohort summary template rendering."""

    def test_render_cohort_summary_with_empty_dict(self):
        """Test render_cohort_summary_template with empty cohort dictionary."""
        html = undertest.render_cohort_summary_template({})

        assert isinstance(html, HTML)
        assert isinstance(html.data, str)

    def test_render_cohort_summary_with_many_cohorts(self):
        """Test render_cohort_summary_template with many cohorts."""
        many_cohorts = {f"cohort_{i}": [f"<table>Data {i}</table>"] for i in range(50)}
        html = undertest.render_cohort_summary_template(many_cohorts)

        assert isinstance(html, HTML)
        # Check that data from various cohorts is rendered
        assert "Data 0" in html.data
        assert "Data 49" in html.data
        assert "<table>" in html.data
