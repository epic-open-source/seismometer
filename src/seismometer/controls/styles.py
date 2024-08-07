from ipywidgets.widgets import HTML, Layout

WIDE_LABEL_STYLE = {"description_width": "120px"}

BOX_GRID_LAYOUT = Layout(
    align_items="flex-start", grid_gap="20px", width="100%", min_width="300px", max_width="1400px"
)
WIDE_BUTTON_LAYOUT = Layout(align_items="flex-start", width="max-content", min_width="200px")


def html_title(title: str) -> HTML:
    """html title style"""
    return HTML(f'<h4 style="text-align: left;  margin: 0px;">{title}</h4>')
