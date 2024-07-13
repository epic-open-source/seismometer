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
