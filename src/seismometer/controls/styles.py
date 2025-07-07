from ipywidgets.widgets import HTML, Layout

WIDE_LABEL_STYLE = {"description_width": "120px"}

BOX_GRID_LAYOUT = Layout(
    align_items="flex-start", grid_gap="20px", width="100%", min_width="300px", max_width="1200px"
)
WIDE_BUTTON_LAYOUT = Layout(align_items="flex-start", width="max-content", min_width="150px", max_width="300px")
WIDE_WIDGET_LAYOUT = Layout(width="max-content", min_width="var(--jp-widgets-inline-label-width)")
DROPDOWN_LAYOUT = Layout(width="calc(max(max-content, var(--jp-widgets-inline-width-short)))")
INLINE_SYMBOL_LAYOUT = Layout(width="10px", align_self="flex-start")
INLINE_LABEL_LAYOUT = Layout(min_width="120px")
TOGGLE_BUTTON_LAYOUT = Layout(flex="1 0 auto")


def html_title(title: str, block: bool = True) -> HTML:
    """html title style"""
    return HTML(f'<h4 style="text-align: left;  margin: 0px;">{title}</h4>', layout=Layout(align_self="flex-start"))


def grid_box_layout(border: bool = False) -> Layout:
    """Return a responsive row-wrap layout with optional border and padding."""
    return Layout(
        display="flex",
        flex_flow="row wrap",
        align_items="flex-start",
        grid_gap="20px",
        border="solid 1px var(--jp-border-color1)" if border else None,
        padding="var(--jp-cell-padding)" if border else None,
    )


def row_wrap_compact_layout(border: bool = False) -> Layout:
    return Layout(
        display="flex",
        flex_flow="row wrap",
        align_items="flex-start",
        grid_gap="3px",
        border="solid 1px var(--jp-border-color1)" if border else None,
        padding="var(--jp-cell-padding)" if border else None,
    )


def vbox_tight_layout(border: bool = False) -> Layout:
    return Layout(
        display="flex",
        flex_flow="column",
        align_items="flex-start",
        grid_gap="4px",
        border="solid 1px var(--jp-border-color1)" if border else None,
        padding="var(--jp-cell-padding)" if border else None,
    )


def vbox_section_layout(border: bool = False) -> Layout:
    return Layout(
        display="flex",
        flex_flow="column",
        align_items="flex-start",
        grid_gap="12px",
        border="solid 1px var(--jp-border-color1)" if border else None,
        padding="var(--jp-cell-padding)" if border else None,
    )
