import pytest

import seismometer.builder.jupyter as undertest


class TestContribCells:
    def test_nocells_returns_none(self):
        assert list(undertest.contrib_cells({"not_cells": True})) == []

    @pytest.mark.parametrize(
        "tags, goodtag",
        [
            pytest.param(["sg-contrib"], True, id="only sg-contrib included"),
            pytest.param(["sg-contrib", "another-tag"], True, id="sg-contrib plus tags is included"),
            pytest.param(["another-tag"], False, id="no sg-contrib plus tags is included"),
        ],
    )
    @pytest.mark.parametrize(
        "celltype, goodtype",
        [
            pytest.param("markdown", True, id="markdown included"),
            pytest.param("code", True, id="code included"),
            pytest.param("raw", True, id="raw included"),
            pytest.param("heading", True, id="heading included"),
            pytest.param("other", False, id="other not included"),
        ],
    )
    def test_cell_types_filter_contrib_cell(self, celltype, goodtype, tags, goodtag):
        onecell = [{"cell_type": celltype, "id": 123456, "metadata": {"tags": tags}}]
        nb = {"cells": onecell}
        assert list(undertest.contrib_cells(nb)) == (onecell if goodtype & goodtag else [])


class TestIdCells:
    @pytest.mark.parametrize(
        "cell, expected",
        [
            pytest.param({"metadata": {"sg-id": "return this"}}, "return this", id="exact key returned string"),
            pytest.param({"metadata": {"sg-id": 1}}, 1, id="exact key return number"),
            pytest.param({"metadata": {"other_key": "dont return this"}}, None, id="no sg-id returns None"),
            pytest.param({"no-metadata": {"sg-id": "dont return this"}}, None, id="No metadata returns None"),
            pytest.param({"metadata": {}}, None, id="empty metadata returns None"),
            pytest.param(
                {"metadata": {"key-1": "val-1", "sg-id": "return this"}, "other": {"sg-id": "not this"}},
                "return this",
                id="return from correct parent and key",
            ),
        ],
    )
    def test_value_returned(self, cell, expected):
        assert undertest.get_id(cell) == expected


class TestGetText:
    @pytest.mark.parametrize(
        "cell, expected",
        [
            pytest.param({"source": "return this"}, "return this", id="exact key returned string"),
            pytest.param({"source": 1}, 1, id="exact key return number"),
            pytest.param({"no-source": "dont return this"}, None, id="no source returns None"),
            pytest.param({"source": ""}, "", id="empty source returns empty string"),
            pytest.param(
                {"source": "return this", "other": "not this"}, "return this", id="return from correct parent and key"
            ),
        ],
    )
    def test_value_returned_with_default(self, cell, expected):
        assert undertest.get_text(cell) == expected

    @pytest.mark.parametrize(
        "cell, strip, expected",
        [
            pytest.param({"source": "return this ✨"}, True, "return this ", id="exact key returned string"),
            pytest.param({"source": "return this ✨"}, False, "return this ✨", id="exact key returned string"),
        ],
    )
    def test_value_with_specified_strip_returns(self, cell, strip, expected):
        assert undertest.get_text(cell, strip_highlight=strip) == expected
