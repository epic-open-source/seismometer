from collections.abc import Iterable
from typing import Optional

import traitlets
from ipywidgets import HTML, Box, Button, Dropdown, Label, Layout, Stack, ToggleButton, ValueWidget, VBox, jslink

from seismometer.configuration.model import CohortHierarchy

from .styles import (
    DROPDOWN_LAYOUT,
    INLINE_LABEL_LAYOUT,
    INLINE_SYMBOL_LAYOUT,
    TOGGLE_BUTTON_LAYOUT,
    WIDE_BUTTON_LAYOUT,
    WIDE_LABEL_STYLE,
    WIDE_WIDGET_LAYOUT,
    grid_box_layout,
    html_title,
    row_wrap_compact_layout,
    vbox_section_layout,
    vbox_tight_layout,
)


class SelectionListWidget(ValueWidget, VBox):
    """
    Vertical list of buttons for selection of a subset of values
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
                layout=TOGGLE_BUTTON_LAYOUT,
            )
            sub_toggle.observe(self._on_button_change, "value")
            self.button_from_option[option] = sub_toggle
            self.option_from_button[sub_toggle] = option

        self.layout = WIDE_WIDGET_LAYOUT
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

    def _update_options(self, new_options: list[str]):
        """
        Update the available options and rebuild the button list while preserving valid selections.
        """
        self.options = tuple(new_options)
        selected_values = tuple(val for val in self.value if val in self.options)
        self.value = selected_values

        # Clear and rebuild button mappings
        self.button_from_option.clear()
        self.option_from_button.clear()

        for option in self.options:
            sub_toggle = ToggleButton(
                value=option in selected_values,
                icon="check" if option in selected_values else "",
                description=str(option),
                tooltip=str(option),
                disabled=self._disabled,
                button_style="",
                layout=TOGGLE_BUTTON_LAYOUT,
            )
            sub_toggle.observe(self._on_button_change, "value")
            self.button_from_option[option] = sub_toggle
            self.option_from_button[sub_toggle] = option

        # Replace children in the button box
        self.button_box.children = list(self.button_from_option.values())

    def get_selection_text(self) -> str:
        """Description of the currently selected values."""
        text = f"{self.title_label.value}: " if self.title_label else ""
        if self.value:
            return text + f"{', '.join([str(x) for x in self.value])}"
        else:
            return text + ""


class HierarchicalSelectionWidget(VBox):
    """
    Widget for displaying and chaining a hierarchy of selection widgets.
    """

    value = traitlets.Dict(help="Current selection across the hierarchy levels")

    def __init__(
        self,
        options: dict[str, tuple],
        hierarchy: CohortHierarchy,
        combinations: "pd.DataFrame",
        values: Optional[dict[str, tuple]] = None,
        border: bool = False,
        show_all: bool = False,
    ):
        """
        Parameters
        ----------
        options : dict[str,tuple]
            Map of column headers to column buttons.
        hierarchy : CohortHierarchy
            Hierarchical structure (e.g. column order).
        combinations : pd.DataFrame
            DataFrame of valid value combinations across the hierarchy.
        values : Optional[dict[str,tuple]], optional
           Values that should be pre-selected, by default None.
        border : bool
            Whether to draw a border around the layout.
        show_all : bool, optional
            If True, show all optoins, else show only selected, by default False.
        """
        self.hierarchy = hierarchy
        self.values = values or {}
        self.options = options
        self._disabled = False
        self.value_update_in_progress = False
        selection_widget_class = SelectionListWidget if show_all else MultiselectDropdownWidget
        self.widgets = {}

        for key in hierarchy.column_order:
            selection_widget = selection_widget_class(
                title=key, options=options.get(key, {}), value=self.values.get(key, None)
            )
            self.widgets[key] = selection_widget
            self.widgets[key].observe(self._on_value_change, "value")
        self.combinations = combinations

        self.value = {}

        # construct the layout with arrows
        label = HTML(f"<b>{hierarchy.name}:</b>", layout=INLINE_LABEL_LAYOUT)
        layout_items = []
        for i, key in enumerate(hierarchy.column_order):
            if key not in self.widgets:
                continue
            layout_items.append(self.widgets[key])
            if i < len(hierarchy.column_order) - 1:
                layout_items.append(HTML("→", layout=INLINE_SYMBOL_LAYOUT))

        hierarchy_row = Box(
            layout_items,
            layout=row_wrap_compact_layout(border),
        )

        layout_box = VBox(
            children=[label, hierarchy_row],
            layout=vbox_tight_layout(border=border),
        )

        super().__init__([layout_box])
        self._on_value_change()
        self.observe(self._on_value_change, "value")

    @property
    def disabled(self):
        return self._disabled

    @disabled.setter
    def disabled(self, value):
        self._disabled = value
        for widget in self.widgets.values():
            widget.disabled = value

    def _on_value_change(self, change=None):
        if self.value_update_in_progress:
            return
        self.value_update_in_progress = True
        self._update_chained_options()
        self.value = {
            k: tuple(self.widgets[k].value)
            for k in self.hierarchy.column_order
            if k in self.widgets and self.widgets[k].value
        }
        self.value_update_in_progress = False

    def _update_chained_options(self):
        """Update child widget options based on upstream selections using hierarchy + combinations."""
        selected = {k: tuple(w.value) for k, w in self.widgets.items() if len(w.value)}

        lvls = self.hierarchy.column_order
        for index in range(len(lvls) - 1):
            parent_lvl = lvls[index]
            child_lvl = lvls[index + 1]

            if parent_lvl not in self.widgets or child_lvl not in self.widgets:
                continue

            child_widget = self.widgets[child_lvl]
            parent_values = selected.get(parent_lvl, ())

            combo_df = self.combinations
            if parent_lvl not in combo_df.columns or child_lvl not in combo_df.columns:
                continue

            if parent_values:
                filtered = combo_df[combo_df[parent_lvl].isin(parent_values)][child_lvl].dropna().unique()
            else:
                parent_options = self.widgets[parent_lvl].options
                filtered = combo_df[combo_df[parent_lvl].isin(parent_options)][child_lvl].dropna().unique()

            new_options = sorted(set(filtered))

            # Update child widget options
            child_widget._update_options(new_options)

            # Filter current value to valid options
            new_value = tuple(val for val in child_widget.value if val in new_options)
            if child_widget.value != new_value:
                child_widget.value = new_value

            # Update selected to pass along to next level
            if child_widget.value:
                selected[child_lvl] = child_widget.value
            elif child_lvl in selected:
                del selected[child_lvl]

    def get_selection_text(self) -> str:
        """Description of selected values across hierarchy levels."""
        parts = []
        for widget in self.widgets.values():
            if widget.value and (text := widget.get_selection_text()):
                parts.append(text)
        return "\n".join(parts)


class FlatSelectionWidget(VBox):
    """
    Widget for displaying and tracking a flat (non-hierarchical) set of selection widgets.
    """

    value = traitlets.Dict(help="Current selection across the flat (non-hierarchical) dimension keys")

    def __init__(
        self,
        options: dict[str, tuple],
        values: Optional[dict[str, tuple]] = None,
        show_all: bool = False,
        border: bool = False,
        title: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        options : dict[str, tuple]
            Mapping from field name to selectable values.
        values : dict[str, tuple], optional
            Mapping of field name to pre-selected values.
        show_all : bool, optional
            Whether to use SelectionListWidget (True) or MultiselectDropdownWidget (False).
        border : bool, optional
            Whether to show a border and padding.
        title : str, optional
            Optional label shown to the left of the widget group, e.g. "Demographics:"
        """
        super().__init__()

        self.options = options
        self.values = values or {}
        self.show_all = show_all
        self._disabled = False
        self.value_update_in_progress = False

        self.selection_widget_class = SelectionListWidget if show_all else MultiselectDropdownWidget
        self.widgets = {}

        # Create selection widgets
        for key, opts in self.options.items():
            widget = self.selection_widget_class(
                title=key,
                options=opts,
                value=self.values.get(key, ()),
            )
            widget.observe(self._on_subselection_change, "value")
            self.widgets[key] = widget

        # Layout all widgets in a flex box
        self.widgets_box = Box(
            children=list(self.widgets.values()),
            layout=grid_box_layout(border=border),
        )

        # Optional label
        self.label = HTML(f"<b>{title}:</b>", layout=INLINE_LABEL_LAYOUT) if title else None

        # Compose final layout
        layout_items = [self.label, self.widgets_box] if self.label else [self.widgets_box]
        layout = VBox(
            layout_items,
            layout=row_wrap_compact_layout(border),
        )

        self.children = [layout]
        self._on_subselection_change()
        self.observe(self._on_value_change, "value")

    @property
    def disabled(self) -> bool:
        return self._disabled

    @disabled.setter
    def disabled(self, disabled: bool):
        self._disabled = disabled
        for widget in self.widgets.values():
            widget.disabled = disabled

    def _on_subselection_change(self, change=None):
        """Update self.value from widgets"""
        if self.value_update_in_progress:
            return
        self.value = {k: tuple(widget.value) for k, widget in self.widgets.items() if widget.value}

    def _on_value_change(self, change=None):
        """Push self.value to widgets"""
        self.value_update_in_progress = True
        for key, widget in self.widgets.items():
            widget.value = self.value.get(key, ())
        self.value_update_in_progress = False

    def get_selection_text(self) -> str:
        """Description of selected values across all flat fields."""
        parts = []
        for widget in self.widgets.values():
            if widget.value and (text := widget.get_selection_text()):
                parts.append(text)
        return "\n".join(parts)


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
        hierarchies: Optional[list[CohortHierarchy]] = None,
        hierarchy_combinations: Optional[dict[tuple[str], "pd.DataFrame"]] = None,
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
        hierarchies: Optional[list[CohortHierarchy]], optional
            List of cohort hierarchies to consider, by default None.
        hierarchy_combinations: Optional[dict[tuple[str], pd.DataFrame]], optional
            Mapping of each hierarchy to valid combinations of values across its levels, by default None.
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
        self.hierarchies = hierarchies or []
        self.hierarchy_combinations = hierarchy_combinations

        self.value_update_in_progress = False
        self.title_box = HTML()

        hierarchy_keys = set()  # all keys used in any hierarchy
        hierarchy_widgets_list = []

        # Step 1: group widgets by hierarchy
        for hierarchy in self.hierarchies:
            visible_keys = [key for key in hierarchy.column_order if key in options]
            hierarchy_keys.update(visible_keys)
            widget_group = HierarchicalSelectionWidget(
                options=options,
                hierarchy=hierarchy,
                values=values,
                combinations=self.hierarchy_combinations[tuple(hierarchy.column_order)],
                border=border,
                show_all=show_all,
            )
            hierarchy_widgets_list.append(widget_group)
            self.selection_widgets.update(widget_group.widgets)
            widget_group.observe(self._on_subselection_change, "value")

        # Step 2: add non-hierarchical widgets
        non_hierarchical_title = "Non-hierarchical selections" if hierarchy_keys else None
        non_hierarchical_options = {key: options[key] for key in options if key not in hierarchy_keys}
        if non_hierarchical_options:
            non_hierarchical_widget_box = FlatSelectionWidget(
                options=non_hierarchical_options,
                values={k: v for k, v in values.items() if k in non_hierarchical_options},
                title=non_hierarchical_title,
                border=border,
                show_all=show_all,
            )
            self.selection_widgets.update(non_hierarchical_widget_box.widgets)
            non_hierarchical_widget_box.observe(self._on_subselection_change, "value")

        widgets_list = (
            hierarchy_widgets_list + [non_hierarchical_widget_box]
            if non_hierarchical_options
            else hierarchy_widgets_list
        )
        self.children = [
            self.title_box,
            VBox(
                children=widgets_list,
                layout=vbox_section_layout(border=border),
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
        if self.value_update_in_progress:
            return
        self.value = {k: tuple(w.value) for k, w in self.selection_widgets.items() if len(w.value)}

    def _on_value_change(self, change=None):
        """Bubble up changes."""
        self.value_update_in_progress = True
        updated_values = {key: self.value.get(key, ()) for key in self.selection_widgets}
        for key, value in updated_values.items():
            self.selection_widgets[key].value = value
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
