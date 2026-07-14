import re

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from IPython.display import SVG
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from seismometer.plot.mpl.likert import _format_count, _plot_counts, _wrap_labels, likert_plot, likert_plot_figure


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {"Disagree": [10, 10], "Neutral": [15, 10], "Agree": [35, 40]}, index=["Likes Cat", "Likes Dog"]
    )


def test_likert_plot_basic(sample_data):
    svg = likert_plot(sample_data)
    assert isinstance(svg, SVG)
    assert len(re.findall(r'id="ax', svg.data)) == 1
    assert "Likert Plot" in svg.data
    assert "Percentages of Responses" in svg.data
    assert "Counts of Each Row" not in svg.data


def test_likert_plot_figure_basic(sample_data):
    fig = likert_plot_figure(sample_data)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert isinstance(ax, Axes)
    assert ax.get_title() == "Likert Plot"
    assert ax.get_xlabel() == "Percentages of Responses"


def test_likert_plot_with_counts(sample_data):
    sample_data_with_diff_sums = sample_data.copy()
    sample_data_with_diff_sums.iloc[1, 0] = 6  # Modify to create different row sums
    svg = likert_plot(sample_data_with_diff_sums)
    assert isinstance(svg, SVG)
    assert len(re.findall(r'id="ax', svg.data)) == 2
    assert "Likert Plot" in svg.data
    assert "Percentages of Responses" in svg.data
    assert "Counts of Each Row" in svg.data


def test_likert_plot_figure_with_counts(sample_data):
    sample_data_with_diff_sums = sample_data.copy()
    sample_data_with_diff_sums.iloc[1, 0] = 6  # Modify to create different row sums
    fig = likert_plot_figure(sample_data_with_diff_sums)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    ax, ax_count = fig.axes
    assert isinstance(ax, Axes)
    assert isinstance(ax_count, Axes)
    assert ax.get_title() == "Likert Plot"
    assert ax_count.get_title() == "Counts of Each Row"


def test_likert_plot_figure_border(sample_data):
    fig = likert_plot_figure(sample_data, border=10)
    ax = fig.axes[0]
    assert ax.get_xlim()[0] <= -10
    assert ax.get_xlim()[1] >= 10


def test_likert_plot_figure_text_annotations(sample_data):
    fig = likert_plot_figure(sample_data)
    ax = fig.axes[0]
    for text in ax.texts:
        assert "%" in text.get_text()
        assert "(" in text.get_text()


def test_plot_counts(sample_data):
    _, ax_count = plt.subplots()
    _plot_counts(sample_data, ax_count)
    assert isinstance(ax_count, Axes)
    assert ax_count.get_title() == "Counts of Each Row"
    assert ax_count.get_xlabel() == "Counts"
    for bar in ax_count.containers[0].get_children():
        assert bar.get_width() in sample_data.sum(axis=1).values


def test_plot_counts_text_annotations(sample_data):
    _, ax_count = plt.subplots()
    _plot_counts(sample_data, ax_count)
    max_count = sample_data.sum(axis=1).max()
    for text in ax_count.texts:
        assert text.get_text().isdigit()
        assert float(text.get_text()) <= max_count


def test_plot_counts_border(sample_data):
    _, ax_count = plt.subplots()
    _plot_counts(sample_data, ax_count, border=10)
    assert ax_count.get_xlim()[1] >= sample_data.sum(axis=1).max() + 10


def test_likert_plot_figure_empty_text(sample_data):
    # Modify sample data to create a scenario where text should be empty
    sample_data_empty_text = sample_data.copy()
    sample_data_empty_text.iloc[0, 0] = 1  # Set a value to 1 to test empty text
    fig = likert_plot_figure(sample_data_empty_text)
    ax = fig.axes[0]
    texts = []
    index = 0
    for bar in ax.patches:
        if bar.get_width() < 5:
            texts.append("")
        else:
            texts.append(ax.texts[index].get_text())
            index = index + 1
    assert len(ax.patches) == len(texts)
    for bar, text in zip(ax.patches, texts):
        if bar.get_width() < 5:
            assert text == ""
        elif bar.get_width() < 9.5:
            assert "%" in text and "(" not in text
        else:
            assert "%" in text and "(" in text


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================


class TestLikertPlotErrorHandling:
    """Test error handling for likert_plot functions."""

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()
        # The error comes from get_balanced_colors when len(df.columns) == 0
        with pytest.raises(ValueError, match="length must be between"):
            likert_plot_figure(empty_df)

    def test_empty_dataframe_likert_plot_raises_error(self):
        """Test that empty DataFrame raises ValueError in likert_plot wrapper."""
        empty_df = pd.DataFrame()
        # The error comes from get_balanced_colors when len(df.columns) == 0
        with pytest.raises(ValueError, match="length must be between"):
            likert_plot(empty_df)


# ============================================================================
# _wrap_labels() EDGE CASES
# ============================================================================


class TestWrapLabels:
    """Test edge cases for _wrap_labels function."""

    @pytest.mark.parametrize(
        "labels,expected",
        [
            # Empty string
            ([""], [""]),
            # Single character
            (["A"], ["A"]),
            # Very long label (should wrap)
            (
                ["ThisIsAVeryLongLabelThatExceedsTheDefaultWidthOf15Characters"],
                ["ThisIsAVeryLong\nLabelThatExceed\nsTheDefaultWidt\nhOf15Characters"],
            ),
            # Label with spaces (wraps at word boundaries)
            (["This is a longer label"], ["This is a\nlonger label"]),
            # Special characters (wraps at word boundaries)
            (["Label with @#$% special chars!"], ["Label with @#$%\nspecial chars!"]),
            # Multiple labels (second one wraps to 3 lines)
            (["Short", "Very Long Label That Should Wrap"], ["Short", "Very Long Label\nThat Should\nWrap"]),
            # Label with newlines (textwrap removes/reorganizes newlines)
            (["Already\nhas\nnewlines"], ["Already has\nnewlines"]),
            # Empty list
            ([], []),
        ],
    )
    def test_wrap_labels_edge_cases(self, labels, expected):
        """Test _wrap_labels with various edge cases."""
        result = _wrap_labels(labels)
        assert result == expected

    def test_wrap_labels_custom_width(self):
        """Test _wrap_labels with custom width."""
        labels = ["This is a long label"]
        result = _wrap_labels(labels, width=5)
        assert result == ["This\nis a\nlong\nlabel"]

    @pytest.mark.parametrize(
        "label,width,expected_lines",
        [
            ("Short", 10, 1),
            ("MediumLength", 5, 3),  # "Mediu" + "mLeng" + "th"
            ("A" * 50, 10, 5),  # 50 characters with width 10 = 5 lines
        ],
    )
    def test_wrap_labels_line_count(self, label, width, expected_lines):
        """Test that wrapping produces expected number of lines."""
        result = _wrap_labels([label], width=width)[0]
        assert result.count("\n") == expected_lines - 1


# ============================================================================
# _format_count() EDGE CASES
# ============================================================================


class TestFormatCount:
    """Test edge cases for _format_count function."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            # Zero
            (0, "0"),
            # Small values
            (1, "1"),
            (99, "99"),
            (999, "999"),
            # Thousands
            (1_000, "1K"),
            (1_234, "1.23K"),
            (9_999, "10K"),
            (10_000, "10K"),
            (999_999, "1e+03K"),  # .3g format produces scientific notation
            # Millions
            (1_000_000, "1M"),
            (1_234_567, "1.23M"),
            (10_000_000, "10M"),
            (999_999_999, "1e+03M"),  # .3g format produces scientific notation
            # Billions
            (1_000_000_000, "1B"),
            (1_234_567_890, "1.23B"),
            (10_000_000_000, "10B"),
            (999_999_999_999, "1e+03B"),  # .3g format produces scientific notation
        ],
    )
    def test_format_count_values(self, value, expected):
        """Test _format_count with various numeric values."""
        result = _format_count(value)
        assert result == expected

    @pytest.mark.parametrize(
        "value,expected_suffix",
        [
            (500, ""),  # No suffix for < 1000
            (5_000, "K"),
            (5_000_000, "M"),
            (5_000_000_000, "B"),
        ],
    )
    def test_format_count_suffix(self, value, expected_suffix):
        """Test that correct suffix is applied."""
        result = _format_count(value)
        if expected_suffix:
            assert result.endswith(expected_suffix)
        else:
            assert not any(result.endswith(s) for s in ["K", "M", "B"])

    def test_format_count_precision(self):
        """Test that formatting maintains reasonable precision."""
        # Test that we get 3 significant figures (via .3g format)
        result = _format_count(1_234_567)
        assert result == "1.23M"

        result = _format_count(9_876_543)
        assert result == "9.88M"


# ============================================================================
# SVG OUTPUT STRUCTURE VALIDATION
# ============================================================================


class TestSVGOutputStructure:
    """Test SVG output structure beyond basic type checking."""

    def test_svg_contains_required_elements(self, sample_data):
        """Test that SVG output contains required XML elements."""
        svg = likert_plot(sample_data)
        svg_str = svg.data

        # Check for essential SVG elements
        assert "<svg" in svg_str, "SVG should contain opening svg tag"
        assert "</svg>" in svg_str, "SVG should contain closing svg tag"
        assert "<g" in svg_str, "SVG should contain group elements"
        assert "<rect" in svg_str or "<path" in svg_str, "SVG should contain rect/path elements (bars)"
        # Matplotlib uses comments for text labels in SVG
        assert "<!--" in svg_str, "SVG should contain text labels (as comments)"

    def test_svg_contains_data_elements(self, sample_data):
        """Test that SVG contains elements representing the data."""
        svg = likert_plot(sample_data)
        svg_str = svg.data

        # Should contain percentage symbols (from bar labels)
        assert svg_str.count("%") >= len(sample_data.index), "SVG should contain percentage labels"

        # Should contain index labels
        for label in sample_data.index:
            assert label in svg_str, f"SVG should contain index label '{label}'"

        # Should contain column labels (legend)
        for col in sample_data.columns:
            assert col in svg_str, f"SVG should contain column label '{col}'"

    def test_svg_structure_with_count_axis(self, sample_data):
        """Test SVG structure when count axis is present."""
        sample_data_diff_sums = sample_data.copy()
        sample_data_diff_sums.iloc[1, 0] = 6  # Different row sums trigger count axis

        svg = likert_plot(sample_data_diff_sums)
        svg_str = svg.data

        # Should have two axes (main plot + count axis)
        assert svg_str.count('id="ax') == 2, "SVG should contain two axes"
        assert "Counts of Each Row" in svg_str, "SVG should contain count axis title"

    @pytest.mark.parametrize("border", [0, 5, 10, 20])
    def test_svg_generation_with_different_borders(self, sample_data, border):
        """Test that SVG is generated successfully with different border values."""
        svg = likert_plot(sample_data, border=border)
        assert isinstance(svg, SVG)
        assert "<svg" in svg.data

    def test_svg_viewbox_present(self, sample_data):
        """Test that SVG has a viewBox attribute for proper scaling."""
        svg = likert_plot(sample_data)
        svg_str = svg.data

        # Check for viewBox or width/height attributes
        assert "viewBox" in svg_str or (
            "width" in svg_str and "height" in svg_str
        ), "SVG should have viewBox or width/height attributes"
