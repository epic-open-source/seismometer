from importlib.resources import files as ir_files
from pathlib import Path

import pytest

import seismometer.configuration.options as undertest

mpp = ir_files("seismometer.builder.resources")
TEMPLATE_DIR = Path(next(mpp.iterdir())).parent
RELEASED_TEMPLATES = ["binary"]


class TestTemplateOptions:
    def test_preconstructed_has_templates(self):
        assert list(undertest.template_options) == RELEASED_TEMPLATES

    def test_default_has_templates(self):
        assert list(undertest.TemplateOptions()) == ["binary"]

    def test_set_over_released_fails(self):
        to = undertest.TemplateOptions()
        with pytest.raises(ValueError, match="Cannot*"):
            to["binary"] = "custom.ipynb"

    def test_missing_get_raises(self):
        to = undertest.TemplateOptions()
        with pytest.raises(AttributeError, match="Template not found*"):
            _ = to["missing"]

    def test_set_other_is_accessible(self):
        to = undertest.TemplateOptions()
        to["NEW"] = "custom.ipynb"
        assert list(to) == RELEASED_TEMPLATES + ["NEW"]

    def test_add_adhoc_adds_custom(self):
        expected = undertest.Option("custom", Path("custom.ipynb"))
        to = undertest.TemplateOptions()

        added = to.add_adhoc_template("custom.ipynb")

        assert list(to) == RELEASED_TEMPLATES + ["custom"]
        assert added == expected

    @pytest.mark.parametrize("key", RELEASED_TEMPLATES)
    def test_can_access_released(self, key):
        expected_resource_dir = TEMPLATE_DIR

        to = undertest.TemplateOptions()

        assert to[key].name == key
        # Assumes install never separates the file and resources
        assert expected_resource_dir in to[key].value.parents

    def test_can_access_added(self):
        to = undertest.TemplateOptions()

        to["test"] = "test.ipynb"
        assert to["test"].name == "test"
        assert to["test"].value == "test.ipynb"
        assert to._custom_templates == {"test": "test.ipynb"}
