from typing import Optional

import traitlets
from ipywidgets import HTML, Box, Dropdown, Label, Layout, Stack, ToggleButton, ValueWidget, VBox, jslink


class SelectionListWidget(ValueWidget, VBox):
    """
    Vertical list of buttons for selection of an subset of values
    """

    value = traitlets.Tuple(help="The selected values for the button list")

    def __init__(self, title: str, options: tuple[str], value: tuple[str] = None, show_title: bool = True):
        """A vertical list of buttons for selection

        Parameters
        ----------
        title : str
            Title to be displayed above the list of buttons.
        options : tuple[str]
            Selectable options.
        value : Optional[tuple[str]], optional
            Subset of options that should be selected by default, by default None.
        show_title : bool, optional
            If True, shows the title, by default True.
        """
        super().__init__()
        self.options = tuple(options)  # make immutable
        selected_values = value or ()
        selected_values = tuple(value for value in selected_values)
        self.buttons = []
        self.value_from_button = {}
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
            self.buttons.append(sub_toggle)
            self.value_from_button[sub_toggle] = option

        self.layout = Layout(width="max-content", min_width="var(--jp-widgets-inline-label-width)")
        self.button_box = VBox(
            children=self.buttons,
            layout=Layout(max_height="calc(7* var(--jp-widgets-inline-height))", align_items="flex-start"),
        )
        self.label = Label(title)
        if show_title:
            self.children = [self.label] + [self.button_box]
        else:
            self.children = [self.button_box]
        self._on_button_change()

    def _on_button_change(self, change=None):
        """Bubble down control change."""
        if change:
            change["owner"].icon = "check" if change["owner"].value else ""
        self.value = self.control_value

    @property
    def title(self) -> str:
        """Selection title."""
        return self.label.value

    @title.setter
    def title(self, title: str):
        self.label.value = title

    @property
    def control_value(self) -> tuple:
        """Value from child widgets."""
        return tuple(self.value_from_button[x] for x in self.buttons if x.value)

    @control_value.setter
    def control_value(self, new_value: tuple[str]):
        """Bubble up changes."""
        for button in self.buttons:
            button.value = button.description in (str(value) for value in new_value)

    def get_selection_text(self) -> str:
        """Descriptoin of the currently selected values."""
        if self.control_value:
            return f"{self.title}: {','.join([str(x) for x in self.control_value])}"
        else:
            return ""


class MultiSelectionListWidget(ValueWidget, VBox):
    """
    Group of selection buttons shown as a collapsale table of options.
    """

    value = traitlets.Dict(help="The selected values for the button lists")

    def __init__(
        self,
        options: dict[str, tuple],
        values: Optional[dict[str, tuple]] = None,
        *,
        title: str = None,
        ghost_text: str = None,
    ):
        """
        A table of buttons organized into columns by thier keys. Collapsable to save space.

        Parameters
        ----------
        options : dict[str,tuple]
            Map of column headers to column buttons.
        values : Optional[dict[str,tuple]], optional
           Values that should be pre-selected, by default None.
        title : str, optional
            Name of the control, by default "".
        ghost_text : str, optional
            What to display when no buttons are selected, by default "Select".
        """
        super().__init__()
        self.title = title
        self.ghost_text = ghost_text
        self.selection_widgets = {}
        if values is None:
            values = {key: () for key in options}
        for key in options:
            selection_widget = SelectionListWidget(title=key, options=options[key], value=values[key])
            self.selection_widgets[key] = selection_widget
            selection_widget.observe(self._on_subselection_change, "value")
        self.title_box = HTML()
        self.children = [
            self.title_box,
            Box(
                children=[self.selection_widgets[key] for key in self.selection_widgets],
                layout=Layout(display="flex", flex_flow="row wrap", align_items="flex-start", grid_gap="20px"),
            ),
        ]
        # self.layout = Layout(width="max-content")
        self.update_title_section(self.title, self.ghost_text)
        self._on_subselection_change()

    def _on_subselection_change(self, change=None):
        """Sets the observable value."""
        self.value = {k: v for k, v in self.control_value.items() if v}

    @property
    def control_value(self) -> dict[str, tuple[str]]:
        """Reads values from child widgets."""
        return {key: self.selection_widgets[key].value for key in self.selection_widgets}

    @control_value.setter
    def control_value(self, new_value):
        """Updates the actual control value."""
        for key in self.selection_widgets:
            self.selection_widgets[key].control_value = new_value.get(key, ())

    def get_selection_text(self) -> str:
        """Return the header text for the widget."""
        selection_strings = []
        for key in self.selection_widgets:
            if selection_text := self.selection_widgets[key].get_selection_text():
                selection_strings.append(selection_text)
        if selection_strings:
            return "\n".join(selection_strings)
        else:
            return self.ghost_text

    def update_title_section(self, title, subtitle):
        if title:
            self.title_box.value = f'<h4 style="text-align: left;  margin: 0px;">{title}</h4>'
            if subtitle:
                self.title_box.value += f"<span>{subtitle}</span>"


class DisjointSelectionListsWidget(ValueWidget, VBox):
    """
    A dropdown which displays a different set of buttons based on the parent option.
    Useful when picking two linked values.
    """

    value = traitlets.Tuple(help="The selected group and its values for the button lists.")

    def __init__(
        self,
        options: dict[str, tuple],
        value: Optional[tuple[str | tuple]] = None,
        *,
        title: str = "",
        select_all: bool = False,
    ):
        """
        A drop down selector where each selection has its own set of buttons.
        The value is the current dropdown, and any of that values selected button values.

        Parameters
        ----------
        options : dict[str,tuple]
            Drown down entires, and thier corresponding buttons.
        value : Optional[tuple[str | tuple]], optional
            Pre-selected values, by default None.
        title : str, optional
            Dispaly above the dropdown, by default "".
        select_all : bool, optional
            As an alternative to value - set all values to selected by default, by default False.
        """
        super().__init__()
        self.title = title
        self.label = Label(title)
        values = {k: options[k] if select_all else () for k in options}
        if value is not None:
            values[value[0]] = value[1]
        else:
            value = (list(values.keys())[0], values[list(values.keys())[0]])
        self.dropdown = Dropdown(
            options=[key for key in values],
            value=value[0],
            layout=Layout(width="max-content", min_width="var(--jp-widgets-inline-label-width)"),
        )
        self.dropdown.observe(self._on_selection_change, "value")
        self.selection_widgets = {}
        for key in options:
            selection_widget = SelectionListWidget(
                title=key, options=options[key], value=values[key], show_title=False
            )
            self.selection_widgets[key] = selection_widget
            selection_widget.observe(self._on_selection_change, "value")
        self.stack = Stack(children=[self.selection_widgets[key] for key in self.selection_widgets], selected_index=0)
        self.children = [self.label, self.dropdown, self.stack]
        jslink((self.dropdown, "index"), (self.stack, "selected_index"))
        self.layout = Layout(width="max-content")
        self._on_selection_change()

    def _on_selection_change(self, *args):
        """Update value from controls."""
        self.value = self.control_value

    @property
    def control_value(self) -> dict[str, tuple[str]]:
        """Read value from child widgets."""
        key = self.dropdown.value
        selections = self.selection_widgets[key].value
        return (key, selections)

    @control_value.setter
    def control_value(self, new_value):
        """Update all the children based on the new value."""
        key = new_value[0]
        value = new_value[1]
        self.selection_widgets[key].control_value = value
        self.dropdown.value = key
