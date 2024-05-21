from collections import namedtuple
from importlib.resources import files as _files
from pathlib import Path

Pathlike = str | Path
Option = namedtuple("Option", ["name", "value"])


# region Template Selection
class TemplateOptions:
    """
    A class to manage the available templates for the builder.

    Acts similar to an Enum, returning Options that have a name and Path value.
    Can be accessed like a dictionary.
    Cannot override the core templates.

    Returns
    -------
    Option
        A named tuple with name and path fields.

    Yields
    ------
    Iterates over all known templates
        Returns names (keys) of the templates.
    """

    _base_templates: dict[str, Path] = {
        "binary": "classifier_bin.ipynb",
    }

    def __init__(self):
        self._custom_templates: dict[str, Path] = {}

    def __getitem__(self, key):
        """Access options directly off instance like a dictionary."""
        if key in TemplateOptions._base_templates:
            import seismometer.builder.resources

            res = _files(seismometer.builder.resources) / self._base_templates[key]
            return Option(key, res)
        if key in self._custom_templates:
            return Option(key, self._custom_templates[key])
        raise AttributeError(f"Template not found: {key}")

    def __setitem__(self, key, value):
        """Set new templates as attributes; cannot override base templates."""
        if key in TemplateOptions._base_templates:
            raise ValueError(f"Cannot override base template: {key}")
        self._custom_templates[key] = value

    def __iter__(self):
        """List all registered templates."""
        yield from TemplateOptions._base_templates.keys()
        yield from self._custom_templates.keys()

    def add_adhoc_template(self, filepath: Pathlike) -> Option:
        """
        Adds a custom template to those known under "custom".

        Parameters
        ----------
        filepath : Pathlike
            The location of the template notebook.

        Returns
        -------
        Option
            The Option tuple for the new template.
        """
        self["custom"] = Path(filepath)
        return self["custom"]


# A global instance of the TemplateOptions
template_options = TemplateOptions()
