import seaborn as sns
import matplotlib.colors as mcolors

def color_to_rgb(color_str: str):
    """
    Converts a color string to an RGB string.

    Parameters
    ----------
    color_str : str
        The input color string. It can be either a hexadecimal color (e.g., "#FF0000") or a color name (e.g., "red").

    Returns
    -------
    str
        The RGB string representation of the input color, formatted as "R, G, B".

    Raises
    ------
    ValueError
        If the input color string is invalid or cannot be converted.
    """
    try:
        # Try to convert a hexadecimal color string to RGB
        if color_str.startswith('#'):
            rgb = mcolors.hex2color(color_str)
        # Try to convert a color name to RGB
        else:
            rgb = mcolors.to_rgb(color_str)
    except ValueError:
        raise ValueError(f"Invalid color name: {color_str}")
    return tuple(int(x * 255) for x in rgb)

def lighten_color(color_str: str, n_colors: int = 3, position: int = 1) -> str:
    """
    Lightens the given color by generating a palette of lighter shades and selecting one based on the specified position.

    Parameters
    ----------
    color_str : str
        The input color string. It can be either a hexadecimal color (e.g., "#FF0000") or a color name (e.g., "red").
    n_colors : int, optional
        The number of colors in the generated palette, by default 5.
    position : int, optional
        The position of the desired lightened color in the palette, by default 1.

    Returns
    -------
    str
        The corresponding lightened color in hexadecimal format.

    Raises
    ------
    ValueError
        If the input color string is invalid or cannot be lightened.
    """
    try:
        color_str_hex = color_str if color_str.startswith("#") else mcolors.to_hex(color_str)
        rgb_normalized = sns.light_palette(color_str_hex,n_colors=n_colors)[position]
        color_hex = mcolors.to_hex(rgb_normalized)
    except ValueError:
        raise ValueError(f"Invalid color name: {color_str}")
    return color_hex

def create_bar(value: float, max_width: int = 75, height: int = 20, color: str = 'green', background_color: str = 'lightgray', opacity: int = 0.5) -> str:
    """
    Create divs to represent `value` as a bar.

    Parameters
    ----------
    value : float
        The proportion to be represented as a bar (between 0 and 1).
    max_width : int, optional
        The maximum width of the bar, by default 75 pxls.
    height : int, optional
        The height of the bar, by default 20 pxls.
    color : str, optional
        The color of the filled portion of the bar, by default 'green'.
    background_color : str, optional
        The background color of the entire bar, by default 'lightgray'.
    opacity : int, optional
        The opacity of the filled portion (between 0 and 1), by default 0.5.

    Returns
    -------
    str
        A string containing HTML div elements representing the bar.
    """
    width = round(max_width * value, 2)
    red,green,blue = color_to_rgb(color_str=color)
    return f"""\
    <div style="width: {max_width}px; background-color:{background_color}; position: relative;">\
        <div style="position: absolute; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; z-index: 1;">\
            <span>{value}</span>\
        </div>\
        <div style="height:{height}px;width:{width}px;background-color:rgba({red}, {green}, {blue}, {opacity}); position: relative; z-index: 2;"></div>\
    </div>\
    """