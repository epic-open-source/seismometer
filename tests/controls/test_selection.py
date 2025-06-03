from unittest.mock import patch

import pandas as pd
import pytest
from ipywidgets import HTML

import seismometer.controls.selection as undertest
from seismometer.configuration.model import CohortHierarchy


class TestSelectionListWidget:
    def test_init(self):
        widget = undertest.SelectionListWidget(options=["a", "b", "c"], value=["a", "b"], title="title")
        assert widget.title_label.value == "title"
        assert widget.value == (
            "a",
            "b",
        )

    def test_init_no_value(self):
        widget = undertest.SelectionListWidget(options=["a", "b", "c"])
        assert widget.title_label is None
        assert widget.value == ()

    def test_value_propagation(self):
        widget = undertest.SelectionListWidget(options=["a", "b", "c"])
        widget.value = (
            "a",
            "b",
        )
        assert widget.button_from_option["a"].value is True
        assert widget.button_from_option["b"].value is True
        assert widget.button_from_option["c"].value is False

    def test_display_text(self):
        widget = undertest.SelectionListWidget(options=["a", "b", "c"], title="test_title")
        widget.value = (
            "a",
            "b",
        )
        assert widget.get_selection_text() == "test_title: a, b"

    def test_disabled(self):
        widget = undertest.SelectionListWidget(options=["a", "b", "c"])
        assert not widget.disabled
        widget.disabled = True
        assert widget.disabled
        assert widget.button_from_option["a"].disabled
        assert widget.button_from_option["b"].disabled
        assert widget.button_from_option["c"].disabled
        widget.disabled = False
        assert not widget.disabled
        assert not widget.button_from_option["a"].disabled
        assert not widget.button_from_option["b"].disabled
        assert not widget.button_from_option["c"].disabled


@pytest.mark.parametrize("show_all", [True, False])
class TestMultiSelectionListWidget:
    @patch("seismometer.seismogram.Seismogram")
    def test_init(self, mock_seismo, show_all):
        fake_seismo = mock_seismo.return_value
        fake_seismo.cohort_hierarchies = []
        widget = undertest.MultiSelectionListWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"]},
            values={"a": ["a", "b"], "b": ["c"]},
            title="title",
            border=True,
            show_all=show_all,
        )
        assert widget.title == "title"
        assert widget.value == {"a": ("a", "b"), "b": ("c",)}

    @patch("seismometer.seismogram.Seismogram")
    def test_init_no_value(self, mock_seismo, show_all):
        fake_seismo = mock_seismo.return_value
        fake_seismo.cohort_hierarchies = []
        widget = undertest.MultiSelectionListWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"]},
            show_all=show_all,
        )
        assert widget.value == {}

    @patch("seismometer.seismogram.Seismogram")
    def test_value_propagation(self, mock_seismo, show_all):
        fake_seismo = mock_seismo.return_value
        fake_seismo.cohort_hierarchies = []
        widget = undertest.MultiSelectionListWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"], "d": ["a", "b", "c"]},
            show_all=show_all,
        )
        assert widget.value == {}
        widget.value = {"a": ("a", "b"), "b": ("c",)}
        assert widget.selection_widgets["a"].value == ("a", "b")
        assert widget.selection_widgets["b"].value == ("c",)
        assert widget.selection_widgets["d"].value == ()

    @patch("seismometer.seismogram.Seismogram")
    def test_on_subselection_changed(self, mock_seismo, show_all):
        fake_seismo = mock_seismo.return_value
        fake_seismo.cohort_hierarchies = []
        widget = undertest.MultiSelectionListWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"], "d": ["a", "b", "c"]},
            show_all=show_all,
        )
        assert widget.value == {}
        widget.selection_widgets["a"].value = ("a", "b")
        assert widget.value == {"a": ("a", "b")}
        assert widget.selection_widgets["b"].value == ()
        assert widget.selection_widgets["d"].value == ()

    @patch("seismometer.seismogram.Seismogram")
    def test_display_text(self, mock_seismo, show_all):
        fake_seismo = mock_seismo.return_value
        fake_seismo.cohort_hierarchies = []
        widget = undertest.MultiSelectionListWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"], "d": ["a", "b", "c"]},
            show_all=show_all,
        )
        assert widget.value == {}
        assert widget.get_selection_text() == ""
        widget.value = {"a": ("a", "b"), "b": ("c",)}
        assert widget.get_selection_text() == "a: a, b\nb: c"

    @patch("seismometer.seismogram.Seismogram")
    def test_disabled(self, mock_seismo, show_all):
        fake_seismo = mock_seismo.return_value
        fake_seismo.cohort_hierarchies = []
        widget = undertest.MultiSelectionListWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"], "d": ["a", "b", "c"]},
            show_all=show_all,
        )
        assert not widget.disabled
        widget.disabled = True
        assert widget.disabled
        assert widget.selection_widgets["a"].disabled
        assert widget.selection_widgets["b"].disabled
        assert widget.selection_widgets["d"].disabled
        widget.disabled = False
        assert not widget.disabled
        assert not widget.selection_widgets["a"].disabled
        assert not widget.selection_widgets["b"].disabled
        assert not widget.selection_widgets["d"].disabled


class TestMultiSelectionListWidgetHierarchyFiltering:
    @patch("seismometer.seismogram.Seismogram")
    def test_parent_filters_child_options(self, mock_seismo):
        fake_seismo = mock_seismo.return_value
        hierarchy = ["level1", "level2"]
        df = pd.DataFrame([("A", "X"), ("A", "Y"), ("B", "Z")], columns=hierarchy)
        fake_seismo.cohort_hierarchies = [CohortHierarchy(name="test", hierarchy=hierarchy)]
        fake_seismo._cohort_hierarchy_combinations = {tuple(hierarchy): df}

        widget = undertest.MultiSelectionListWidget(
            options={"level1": ["A", "B", "C"], "level2": ["X", "Y", "Z"]},
            values={},
        )

        widget.selection_widgets["level1"].value = ("A",)
        assert sorted(widget.selection_widgets["level2"].options) == ["X", "Y"]

    @patch("seismometer.seismogram.Seismogram")
    def test_child_shows_all_if_no_parent_selected(self, mock_seismo):
        fake_seismo = mock_seismo.return_value
        hierarchy = ["p", "c"]
        df = pd.DataFrame([("X", "1"), ("Y", "2"), ("Z", "3")], columns=hierarchy)
        fake_seismo.cohort_hierarchies = [CohortHierarchy(name="test", hierarchy=hierarchy)]
        fake_seismo._cohort_hierarchy_combinations = {tuple(hierarchy): df}

        widget = undertest.MultiSelectionListWidget(
            options={"p": ["X", "Y", "Z"], "c": ["1", "2", "3"]},
            values={},
        )

        widget.selection_widgets["p"].value = ()  # No parent selected
        assert sorted(widget.selection_widgets["c"].options) == ["1", "2", "3"]

    @patch("seismometer.seismogram.Seismogram")
    def test_child_value_dropped_if_invalid(self, mock_seismo):
        fake_seismo = mock_seismo.return_value
        hierarchy = ["region", "site"]
        df = pd.DataFrame([("North", "Alpha"), ("South", "Beta")], columns=hierarchy)
        fake_seismo.cohort_hierarchies = [CohortHierarchy(name="test", hierarchy=hierarchy)]
        fake_seismo._cohort_hierarchy_combinations = {tuple(hierarchy): df}

        widget = undertest.MultiSelectionListWidget(
            options={"region": ["North", "South"], "site": ["Alpha", "Beta", "Gamma"]},
            values={},
        )

        widget.selection_widgets["site"].value = ("Beta", "Gamma")
        widget.selection_widgets["region"].value = ("South",)
        assert widget.selection_widgets["site"].value == ("Beta",)

    @patch("seismometer.seismogram.Seismogram")
    def test_first_level_update_cascades_to_second_level(self, mock_seismo):
        fake_seismo = mock_seismo.return_value
        hierarchy = ["lvl1", "lvl2"]
        df = pd.DataFrame([("A", "x1"), ("A", "x2"), ("B", "y1")], columns=hierarchy)
        fake_seismo.cohort_hierarchies = [CohortHierarchy(name="test", hierarchy=hierarchy)]
        fake_seismo._cohort_hierarchy_combinations = {tuple(hierarchy): df}

        widget = undertest.MultiSelectionListWidget(
            options={"lvl1": ["A", "B"], "lvl2": ["x1", "x2", "y1", "z1"]},
            values={},
        )

        widget.selection_widgets["lvl2"].value = ("x1", "y1")
        widget.selection_widgets["lvl1"].value = ("B",)
        assert sorted(widget.selection_widgets["lvl2"].options) == ["y1"]
        assert widget.selection_widgets["lvl2"].value == ("y1",)

    @patch("seismometer.seismogram.Seismogram")
    def test_three_level_hierarchy_filters_correctly(self, mock_seismo):
        fake_seismo = mock_seismo.return_value
        hierarchy = ["lvl1", "lvl2", "lvl3"]
        df = pd.DataFrame(
            [
                ("A", "X", "i"),
                ("A", "X", "j"),
                ("A", "Y", "k"),
                ("B", "Z", "l"),
            ],
            columns=hierarchy,
        )
        fake_seismo.cohort_hierarchies = [CohortHierarchy(name="3lvl", hierarchy=hierarchy)]
        fake_seismo._cohort_hierarchy_combinations = {tuple(hierarchy): df}

        widget = undertest.MultiSelectionListWidget(
            options={"lvl1": ["A", "B"], "lvl2": ["X", "Y", "Z"], "lvl3": ["i", "j", "k", "l"]},
            values={},
        )

        widget.selection_widgets["lvl2"].value = ("X",)
        widget.selection_widgets["lvl1"].value = ("A",)
        assert sorted(widget.selection_widgets["lvl3"].options) == ["i", "j"]

    @patch("seismometer.seismogram.Seismogram")
    def test_hierarchy_with_multiple_visible_keys_renders_correct_arrow_count(self, mock_seismo):
        fake_seismo = mock_seismo.return_value
        hierarchy = CohortHierarchy(name="demo", hierarchy=["level1", "level2", "level3"])
        fake_seismo.cohort_hierarchies = [hierarchy]
        fake_seismo._cohort_hierarchy_combinations = {}

        widget = undertest.MultiSelectionListWidget(
            options={
                "level1": ["A"],
                "level2": ["B"],
                "level3": ["C"],
            },
            values={"level1": ["A"], "level2": ["B"], "level3": ["C"]},
            hierarchies=[hierarchy],
            show_all=True,
        )

        hierarchy_vbox = widget.children[1]  # VBox of hierarchy + non-hierarchy
        hierarchy_boxes = hierarchy_vbox.children[0:-1]  # exclude non-hierarchical box

        for box in hierarchy_boxes:
            arrow_widgets = [child for child in box.children if isinstance(child, HTML) and child.value == "→"]
            assert (
                len(arrow_widgets) == len(hierarchy.hierarchy) - 1
            ), f"Expected {len(hierarchy.hierarchy) - 1} arrows, found {len(arrow_widgets)}"


class TestDisjointSelectionListsWidget:
    def test_init(self):
        widget = undertest.DisjointSelectionListsWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"]},
            value=("a", ["a", "b"]),
            title="title",
        )
        assert widget.title == "title"
        assert widget.value == ("a", ("a", "b"))
        assert len(widget.selection_widgets) == 2
        assert widget.selection_widgets["a"].value == ("a", "b")
        assert widget.selection_widgets["b"].value == (
            "a",
            "c",
        )

    def test_init_no_value(self):
        widget = undertest.DisjointSelectionListsWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"]},
            title="title",
        )
        assert widget.title == "title"
        assert widget.value == ("a", ("a", "b", "c"))
        assert len(widget.selection_widgets) == 2
        assert widget.selection_widgets["a"].value == ("a", "b", "c")
        assert widget.selection_widgets["b"].value == (
            "a",
            "c",
        )

    def test_value_change(self):
        widget = undertest.DisjointSelectionListsWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"]},
            value=("a", ["a", "b"]),
            title="title",
        )
        assert widget.value == ("a", ("a", "b"))
        widget.value = ("b", ("c",))
        assert widget.selection_widgets["a"].value == ("a", "b")
        assert widget.selection_widgets["b"].value == ("c",)
        assert widget.value == ("b", ("c",))

    def test_display_text(self):
        widget = undertest.DisjointSelectionListsWidget(
            options={"a": ["a1", "a2", "a3"], "b": ["b1", "b2"]},
            value=("a", ["a1", "a3"]),
            title="title",
        )
        assert widget.get_selection_text() == "a: a1, a3"

    def test_disabled(self):
        widget = undertest.DisjointSelectionListsWidget(
            options={"a": ["a1", "a2", "a3"], "b": ["b1", "b2"]},
            value=("a", ["a1", "a3"]),
            title="title",
        )
        assert not widget.disabled
        widget.disabled = True
        assert widget.disabled
        assert widget.dropdown.disabled
        assert widget.selection_widgets["a"].disabled
        assert widget.selection_widgets["b"].disabled
        widget.disabled = False
        assert not widget.disabled
        assert not widget.dropdown.disabled
        assert not widget.selection_widgets["a"].disabled
        assert not widget.selection_widgets["b"].disabled

    def test_disable_dropdown_only_one_group(self):
        widget = undertest.DisjointSelectionListsWidget(
            options={"a": ["a1", "a2", "a3"]},
            value=("a", ["a1", "a2"]),
            title="title",
        )
        assert not widget.disabled
        widget.disabled = True
        assert widget.disabled
        assert widget.dropdown.disabled
        assert widget.selection_widgets["a"].disabled
        widget.disabled = False
        assert not widget.disabled
        assert widget.dropdown.disabled
        assert not widget.selection_widgets["a"].disabled


class TestMultiselectDropdownWidget:
    def test_init(self):
        widget = undertest.MultiselectDropdownWidget(options=["a", "b", "c"], value=["a", "b"], title="title")
        assert widget.title == "title"
        assert widget.value == ("a", "b")
        assert widget.get_selection_text() == "title: a, b"

    def test_init_no_value(self):
        widget = undertest.MultiselectDropdownWidget(options=["a", "b", "c"])
        assert widget.title is None
        assert widget.value == ()
        assert widget.get_selection_text() == ""

    def test_add_all(self):
        widget = undertest.MultiselectDropdownWidget(options=["a", "b", "c"])
        widget.dropdown.value = -1
        assert widget.value == ("a", "b", "c")
        assert widget.dropdown.value == -2  # reset to default

    def test_value_propagation(self):
        widget = undertest.MultiselectDropdownWidget(options=["a", "b", "c"])
        widget.value = ("a", "b")
        assert len(widget.selection_options.children) == 2
        assert widget.selection_options.children[0].description == "a"
        assert widget.selection_options.children[1].description == "b"

    def test_value_propagation_remove_option(self):
        widget = undertest.MultiselectDropdownWidget(options=["a", "b", "c"], value=["a", "b"], title="title")
        assert widget.title == "title"
        widget.selection_options.children[0].click()
        assert widget.value == ("b",)

    def test_dropdown_changes_value(self):
        widget = undertest.MultiselectDropdownWidget(options=["a", "b", "c"])
        widget.dropdown.value = 1  # index of b
        assert widget.value == ("b",)
        widget.dropdown.value = 2  # index of c
        assert widget.value == (
            "b",
            "c",
        )
        widget.dropdown.value = 0  # index of a
        assert widget.value == (
            "a",
            "b",
            "c",
        )

    def test_disabled(self):
        widget = undertest.MultiselectDropdownWidget(options=["a", "b", "c"])
        assert not widget.disabled
        widget.disabled = True
        assert widget.disabled
        assert widget.dropdown.disabled
        widget.disabled = False
        assert not widget.disabled
        assert not widget.dropdown.disabled
