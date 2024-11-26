import pytest

import seismometer.controls.selection as undertest


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
    def test_init(self, show_all):
        widget = undertest.MultiSelectionListWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"]},
            values={"a": ["a", "b"], "b": ["c"]},
            title="title",
            border=True,
            show_all=show_all,
        )
        assert widget.title == "title"
        assert widget.value == {"a": ("a", "b"), "b": ("c",)}

    def test_init_no_value(self, show_all):
        widget = undertest.MultiSelectionListWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"]},
            show_all=show_all,
        )
        assert widget.value == {}

    def test_value_propagation(self, show_all):
        widget = undertest.MultiSelectionListWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"], "d": ["a", "b", "c"]},
            show_all=show_all,
        )
        assert widget.value == {}
        widget.value = {"a": ("a", "b"), "b": ("c",)}
        assert widget.selection_widgets["a"].value == ("a", "b")
        assert widget.selection_widgets["b"].value == ("c",)
        assert widget.selection_widgets["d"].value == ()

    def test_on_subselection_changed(self, show_all):
        widget = undertest.MultiSelectionListWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"], "d": ["a", "b", "c"]},
            show_all=show_all,
        )
        assert widget.value == {}
        widget.selection_widgets["a"].value = ("a", "b")
        assert widget.value == {"a": ("a", "b")}
        assert widget.selection_widgets["b"].value == ()
        assert widget.selection_widgets["d"].value == ()

    def test_display_text(self, show_all):
        widget = undertest.MultiSelectionListWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"], "d": ["a", "b", "c"]},
            show_all=show_all,
        )
        assert widget.value == {}
        assert widget.get_selection_text() == ""
        widget.value = {"a": ("a", "b"), "b": ("c",)}
        assert widget.get_selection_text() == "a: a, b\nb: c"

    def test_disabled(self, show_all):
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
