from collections.abc import Iterable
from typing import Optional

import traitlets
from ipywidgets import HTML, Box, Button, Dropdown, Label, Layout, Stack, ToggleButton, ValueWidget, VBox, jslink

from .styles import DROPDOWN_LAYOUT, WIDE_BUTTON_LAYOUT, WIDE_LABEL_STYLE, html_title


class SelectionListWidget(ValueWidget, VBox):
    """
    Vertical list of buttons for selection of an subset of values
    """

    value = traitlets.Tuple(help="The selected values for the button list")

    def __init__(self, options: Iterable[str], *, value: Optional[Iterable[str]] = None, title: Optional[str] = None):
        """A vertical list of buttons for selection

        Parameters
        ----------
        options : Iterable[str]
            Selectable options.
        value : Optional[Iterable[str]], optional
            Subset of options that should be selected by default, by default None.
        title : Optional[str, optional]
            Title to be displayed above the list of buttons, if not set, title not included
        """
        super().__init__()
        self.options = tuple(options)  # make immutable
        selected_values = tuple(value) if value else ()
        self.value = selected_values
        self.button_from_option = {}
        self.option_from_button = {}
        for option in self.options:
            sub_toggle = ToggleButton(
                value=option in selected_values,
                icon="check" if option in selected_values else "",
                description=str(option),
                tooltip=str(option),
                disabled=False,
                button_style="",
                layout=Layout(flex="1 0 auto"),
            )
            sub_toggle.observe(self._on_button_change, "value")
            self.button_from_option[option] = sub_toggle
            self.option_from_button[sub_toggle] = option

        self.layout = Layout(width="max-content", min_width="var(--jp-widgets-inline-label-width)")
        self.button_box = VBox(
            children=[button for button in self.button_from_option.values()],
            layout=Layout(max_height="calc(7* var(--jp-widgets-inline-height))", align_items="flex-start"),
        )
        if title:
            self.title_label = Label(title)
            self.children = [self.title_label] + [self.button_box]
        else:
            self.title_label = None
            self.children = [self.button_box]
        self.observe(self._on_value_change, "value")
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        for widget in self.button_from_option.values():
            widget.disabled = disabled

    def _on_button_change(self, change=None):
        """Bubble down control change."""
        if change:
            change["owner"].icon = "check" if change["owner"].value else ""
        self.value = tuple(option for button, option in self.option_from_button.items() if button.value)

    def _on_value_change(self, change=None):
        """Bubble up changes."""
        updated_values = {option: (option in self.value) for option in self.options}
        for option, value in updated_values.items():
            self.button_from_option[option].value = value

    def get_selection_text(self) -> str:
        """Description of the currently selected values."""
        text = f"{self.title_label.value}: " if self.title_label else ""
        if self.value:
            return text + f"{', '.join([str(x) for x in self.value])}"
        else:
            return text + ""


class MultiSelectionListWidget(ValueWidget, VBox):
    """
    Group of selection buttons shown as a collapsable table of options.
    """

    value = traitlets.Dict(help="The selected values for the button lists")

    def __init__(
        self,
        options: dict[str, tuple],
        values: Optional[dict[str, tuple]] = None,
        *,
        title: str = None,
        border: bool = False,
        show_all: bool = False,
    ):
        """
        A table of buttons organized into columns by their keys. Collapsable to save space.

        Parameters
        ----------
        options : dict[str,tuple]
            Map of column headers to column buttons.
        values : Optional[dict[str,tuple]], optional
           Values that should be pre-selected, by default None.
        title : str, optional
            Name of the control, by default None.
        border : bool, optional
            If True, display a border around the widget, by default False.
        show_all : bool, optional
            If True, show all optoins, else show only selected, by default False.
        """
        super().__init__()
        self.title = title
        self.options = options
        self.selection_widgets = {}
        if values is None:
            values = {}
        else:
            values = {k: tuple(v) for k, v in values.items()}
        self.value = values

        selection_widget_class = SelectionListWidget if show_all else MultiselectDropdownWidget
        for key in options:
            selection_widget = selection_widget_class(title=key, options=options[key], value=values.get(key, None))
            self.selection_widgets[key] = selection_widget
            selection_widget.observe(self._on_subselection_change, "value")
        self.value_update_in_progress = False
        self.title_box = HTML()
        self.children = [
            self.title_box,
            Box(
                children=[self.selection_widgets[key] for key in self.selection_widgets],
                layout=Layout(
                    display="flex",
                    flex_flow="row wrap",
                    align_items="flex-start",
                    grid_gap="20px",
                    border="solid 1px var(--jp-border-color1)" if border else None,
                    padding="var(--jp-cell-padding)" if border else None,
                ),
            ),
        ]
        self.update_title_section(self.title)
        self.observe(self._on_value_change, "value")
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        for widget in self.selection_widgets.values():
            widget.disabled = disabled

    def _on_subselection_change(self, change=None):
        """Sets the observable value."""
        if self.value_update_in_progress:
            return
        if change and change["owner"] in self.selection_widgets.values():
            self.value = {k: tuple(v.value) for k, v in self.selection_widgets.items() if len(v.value)}

    def _on_value_change(self, change=None):
        """Bubble up changes."""
        self.value_update_in_progress = True
        updated_values = {key: self.value.get(key, ()) for key in self.selection_widgets}
        for key, value in updated_values.items():
            self.selection_widgets[key].value = self.value.get(key, ())
        self.value_update_in_progress = False

    def get_selection_text(self) -> str:
        """Return the header text for the widget."""
        selection_strings = []
        for key in self.value:
            if selection_text := self.selection_widgets[key].get_selection_text():
                selection_strings.append(selection_text)
        if selection_strings:
            return "\n".join(selection_strings)
        else:
            return ""

    def update_title_section(self, title):
        if title:
            self.title_box.value = html_title(title).value


class DisjointSelectionListsWidget(ValueWidget, VBox):
    """
    A dropdown which displays a different set of buttons based on the parent option.
    Useful when picking two linked values.
    """

    value = traitlets.Tuple(help="The selected group and its values for the button lists.")

    def __init__(
        self,
        options: dict[str, tuple],
        value: Optional[tuple[str, tuple]] = None,
        *,
        title: str = "",
        select_all: bool = True,
    ):
        """
        A drop down selector where each selection has its own set of buttons.
        The value is the current dropdown, and any of that values selected button values.

        Parameters
        ----------
        options : dict[str,tuple]
            Dropdown entires, and their corresponding buttons.
        value : Optional[tuple[str, tuple]], optional
            Pre-selected values, by default None.
        title : str, optional
            Display above the dropdown, by default "".
        select_all : bool, optional
            As an alternative to value - set all values to selected by default, by default True.
        """
        super().__init__()
        self.title = title
        self.title_box = html_title(title)

        # preselect all values if select_all is set, will override with values next
        values = {k: options[k] if select_all else () for k in options}

        if value is not None:
            # We have a value (cohort/subgroups) so set the dropdown/selection list accordingly
            values[value[0]] = value[1]
        else:
            # No value, so default ot the first key/value pair in values
            dropdown_value = tuple(values.keys())[0]  # first key
            selection_value = values[dropdown_value]
            value = (dropdown_value, selection_value)

        self.dropdown = Dropdown(
            options=[key for key in values], value=value[0], layout=DROPDOWN_LAYOUT, disabled=len(values) == 1
        )

        self.dropdown.observe(self._on_selection_change, "value")
        self.selection_widgets = {}
        for key in options:
            selection_widget = SelectionListWidget(options=options[key], value=values[key])
            self.selection_widgets[key] = selection_widget
            selection_widget.observe(self._on_selection_change, "value")
        self.stack = Stack(children=[self.selection_widgets[key] for key in self.selection_widgets], selected_index=0)
        self.children = [self.title_box, self.dropdown, self.stack]
        jslink((self.dropdown, "index"), (self.stack, "selected_index"))
        self.layout = Layout(width="calc(100% - var(--jp-widgets-border-width)* 2)", max_width="min-content")
        self._on_selection_change()
        self.observe(self._on_value_change, "value")
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        self.dropdown.disabled = len(self.dropdown.options) == 1 or disabled
        for widget in self.selection_widgets.values():
            widget.disabled = disabled

    def _on_selection_change(self, *args):
        """Update value from controls."""
        key = self.dropdown.value
        selections = self.selection_widgets[key].value
        self.value = (key, tuple(selections))

    def _on_value_change(self, change=None):
        """Bubble up changes."""
        key, value = self.value
        self.dropdown.value = key
        self.selection_widgets[key].value = value

    def get_selection_text(self) -> str:
        """Return the selection for the widget as a key value pair."""
        key = self.value[0]
        return f"{key}: {self.selection_widgets[key].get_selection_text()}"


class MultiselectDropdownWidget(ValueWidget, VBox):
    """
    A multi select dropdown which allows multiple selections from a dropdown which get displayed as toggle buttons.
    """

    value = traitlets.Tuple(help="The selected values for the dropdown.")

    def __init__(
        self,
        options: tuple[str],
        value: Optional[tuple[str]] = None,
        *,
        title: str = None,
    ):
        """
        A dropdown selector where multiple selections are allowed.

        Parameters
        ----------
        options : tuple[str]
            Dropdown entries.
        value : Optional[tuple[str]], optional
            Pre-selected values, by default None.
        title : str, optional
            Display show in the dropdown, by default "Add...".
        """
        self.options = tuple(options)
        self.value = tuple(value) if value else ()
        self.title = title
        default_option = (title, -2) if title else ("Add...", -2)
        add_all_option = ("Add all", -1)
        tooltip = f"Select an option to add to {title}" if title else "Select an option to add"
        self.dropdown = Dropdown(
            options=[default_option, add_all_option] + [(str(v), i) for i, v in enumerate(options)],
            index=0,
            style=WIDE_LABEL_STYLE,
            tooltip=tooltip,
            layout=WIDE_BUTTON_LAYOUT,
        )
        self.selection_options = VBox(children=[], layout=Layout(align_self="flex-end", align_items="flex-end"))
        self.buttons = {
            option: Button(
                description=str(option),
                tooltip=f"Remove {option}",
                button_style="primary",
                layout=WIDE_BUTTON_LAYOUT,
            )
            for option in options
        }

        for val in self.value:
            self._insert_button(val)

        children = [self.dropdown, self.selection_options]
        super().__init__(children=children, layout=Layout(width="min-content"))

        for button in self.buttons.values():
            button.on_click(self._remove_button)
        self.dropdown.observe(self._on_dropdown_changed, "value")
        self.observe(self._on_value_change, "value")
        self._disabled = False

    def _remove_button(self, button, update_value=True):
        self.selection_options.children = [child for child in self.selection_options.children if child != button]
        if update_value:
            self.value = tuple(val for val in self.value if self.buttons[val] != button)

    def _insert_button(self, val):
        button = self.buttons[val]
        if button not in self.selection_options.children:
            self.selection_options.children = [child for child in self.selection_options.children] + [button]

    def _on_dropdown_changed(self, change):
        if change["owner"] != self.dropdown:
            return
        option_index = change["new"]
        match option_index:
            case -2:
                pass
            case -1:
                self.value = self.options
            case option_index if self.options[option_index] in self.value:
                pass
            case _:
                self.value = tuple(
                    [value for value in self.options if value in self.value or value == self.options[option_index]]
                )

        self.selection_options.children = [self.buttons[val] for val in self.value]
        self.dropdown.value = -2

    def _update_options(self, new_options: list[str]):
        self.options = tuple(new_options)
        self.buttons = {
            option: Button(
                description=str(option),
                tooltip=f"Remove {option}",
                button_style="primary",
                layout=WIDE_BUTTON_LAYOUT,
            )
            for option in self.options
        }

        # Update dropdown entries
        default_option = (self.title or "Add...", -2)
        add_all_option = ("Add all", -1)
        self.dropdown.options = [default_option, add_all_option] + [(str(v), i) for i, v in enumerate(self.options)]

        # Re-bind events
        for button in self.buttons.values():
            button.on_click(self._remove_button)

        # Reset current selection
        valid_values = [v for v in self.value if v in self.options]
        self.value = tuple(valid_values)
        self._on_value_change({"new": self.value})

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        self.dropdown.disabled = disabled
        for button in self.buttons.values():
            button.disabled = disabled

    def _on_value_change(self, change=None):
        """Bubble up changes."""
        self.selection_options.children = [self.buttons[val] for val in change["new"]]
        self.dropdown.value = -2

    def get_selection_text(self) -> str:
        """Description of the currently selected values."""
        text = f"{self.title}: " if self.title else ""
        if self.value:
            return text + f"{', '.join([str(x) for x in self.value])}"
        else:
            return text + ""
