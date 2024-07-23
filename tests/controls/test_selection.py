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


class TestMultiSelectionListWidget:
    def test_init(self):
        widget = undertest.MultiSelectionListWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"]},
            values={"a": ["a", "b"], "b": ["c"]},
            title="title",
            border=True,
        )
        assert widget.title == "title"
        assert widget.value == {"a": ("a", "b"), "b": ("c",)}

    def test_init_no_value(self):
        widget = undertest.MultiSelectionListWidget(options={"a": ["a", "b", "c"], "b": ["a", "c"]})
        assert widget.value == {}

    def test_value_propagation(self):
        widget = undertest.MultiSelectionListWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"], "d": ["a", "b", "c"]},
        )
        assert widget.value == {}
        widget.value = {"a": ("a", "b"), "b": ("c",)}
        assert widget.selection_widgets["a"].value == ("a", "b")
        assert widget.selection_widgets["b"].value == ("c",)
        assert widget.selection_widgets["d"].value == ()

    def test_on_subselection_changed(self):
        widget = undertest.MultiSelectionListWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"], "d": ["a", "b", "c"]},
        )
        assert widget.value == {}
        widget.selection_widgets["a"].value = ("a", "b")
        assert widget.value == {"a": ("a", "b")}
        assert widget.selection_widgets["b"].value == ()
        assert widget.selection_widgets["d"].value == ()

    def test_display_text(self):
        widget = undertest.MultiSelectionListWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"], "d": ["a", "b", "c"]},
        )
        assert widget.value == {}
        assert widget.get_selection_text() == ""
        widget.value = {"a": ("a", "b"), "b": ("c",)}
        assert widget.get_selection_text() == "a: a, b\nb: c"

    def test_disabled(self):
        widget = undertest.MultiSelectionListWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"], "d": ["a", "b", "c"]},
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
            options={"a": ["a", "b", "c"], "b": ["a", "c"]},
            value=("a", ["a", "b"]),
            title="title",
        )
        assert widget.get_selection_text() == "a: a, b"

    def test_disabled(self):
        widget = undertest.DisjointSelectionListsWidget(
            options={"a": ["a", "b", "c"], "b": ["a", "c"]},
            value=("a", ["a", "b"]),
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
