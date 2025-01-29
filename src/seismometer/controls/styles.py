from ipywidgets.widgets import HTML, Layout

WIDE_LABEL_STYLE = {"description_width": "120px"}

BOX_GRID_LAYOUT = Layout(
    align_items="flex-start", grid_gap="20px", width="100%", min_width="300px", max_width="1200px"
)
WIDE_BUTTON_LAYOUT = Layout(align_items="flex-start", width="max-content", min_width="150px", max_width="300px")
DROPDOWN_LAYOUT = Layout(width="calc(max(max-content, var(--jp-widgets-inline-width-short)))")


def html_title(title: str, block: bool = True) -> HTML:
    """html title style"""
    return HTML(f'<h4 style="text-align: left;  margin: 0px;">{title}</h4>', layout=Layout(align_self="flex-start"))
