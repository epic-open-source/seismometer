import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from seismometer.plot.mpl.likert import _plot_counts, likert_plot


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {"Disagree": [10, 5], "Neutral": [15, 10], "Agree": [35, 45]}, index=["Likes Cat", "Likes Dog"]
    )


def test_likert_plot_basic(sample_data):
    fig = likert_plot(sample_data)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert isinstance(ax, Axes)
    assert ax.get_title() == "Likert Plot"
    assert ax.get_xlabel() == "Percentages of Responses"


def test_likert_plot_with_counts(sample_data):
    sample_data_with_diff_sums = sample_data.copy()
    sample_data_with_diff_sums.iloc[1, 0] = 6  # Modify to create different row sums
    fig = likert_plot(sample_data_with_diff_sums)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2
    ax, ax_count = fig.axes
    assert isinstance(ax, Axes)
    assert isinstance(ax_count, Axes)
    assert ax.get_title() == "Likert Plot"
    assert ax_count.get_title() == "Counts of Each Row"


def test_likert_plot_colors(sample_data):
    custom_colors = ["#FF0000", "#00FF00", "#0000FF"]
    fig = likert_plot(sample_data, colors=custom_colors)
    ax = fig.axes[0]
    for i, bar_container in enumerate(ax.containers):
        for bar in bar_container.get_children():
            assert bar.get_facecolor() == to_rgba(custom_colors[i])


def test_likert_plot_border(sample_data):
    fig = likert_plot(sample_data, border=10)
    ax = fig.axes[0]
    assert ax.get_xlim()[0] <= -10
    assert ax.get_xlim()[1] >= 10


def test_likert_plot_text_annotations(sample_data):
    fig = likert_plot(sample_data)
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


def test_likert_plot_empty_text(sample_data):
    # Modify sample data to create a scenario where text should be empty
    sample_data_empty_text = sample_data.copy()
    sample_data_empty_text.iloc[0, 0] = 1  # Set a value to 1 to test empty text
    fig = likert_plot(sample_data_empty_text)
    ax = fig.axes[0]
    texts = []
    index = 0
    for bar in ax.patches:
        if bar.get_width() < 10:
            texts.append("")
        else:
            texts.append(ax.texts[index].get_text())
            index = index + 1
    assert len(ax.patches) == len(texts)
    for bar, text in zip(ax.patches, texts):
        if bar.get_width() < 10:
            assert text == ""
        else:
            assert text != ""
