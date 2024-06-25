from collections import namedtuple
from importlib.resources import files as _files
from itertools import cycle

import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

# Map of non-semantic colors
NonSemanticColors = namedtuple(
    "NonSemanticColors",
    [
        "Azure",
        "Fuchsia",
        "Lime",
        "Dahlia",
        "Caribbean",
        "Tangerine",
        "Wisteria",
        "Mediterranean",
        "Blush",
        "Sky",
        "Rose",
    ],
)

IOColors = namedtuple("IOColors", ["Camel", "Golden"])

NeutralColors = namedtuple("NeutralColors", ["Light", "Medium", "Dark"])

SemanticColors = namedtuple(
    "SemanticColors",
    [
        "PositiveEffectsDirect",
        "PositiveEffectsIndirect",
        "CausesSpecific",
        "CausesGeneral",
        "ToDoOngoing",
        "ToDoImmediate",
        "AdminOngoing",
        "AdminCommunication",
        "AdminFinancial",
        "DocumentationSubjective",
        "DocumentationObjective",
        "Neutral",
        "NeutralThemed",
        "DoNotUse",
    ],
)

AlertColors = namedtuple("AlertColors", ["Alarm", "Important", "Warning", "Positive", "PositiveLight", "Normal"])

OpacityLevels = namedtuple("OpacityLevels", ["Transparent", "Low", "Medium", "High", "Opaque"])
opacity_levels = OpacityLevels(0, 0.1, 0.2, 0.4, 1)

alert_colors = AlertColors(
    "#c70000",  # Red - alarm
    "#ff6d00",  # Orange - important warning
    "#ffba00",  # Yellow - warning
    "#90da8b",  # Green - positive
    "#caeec8",  # Light Green - positive, low-intensity
    "#0085f2",  # Blue - normal
)

# Slight variation exists for line, area, and text
line_colors = NonSemanticColors(
    "#006bdf",  # Azure
    "#fd00ab",  # Fuchsia
    "#229d00",  # Lime
    "#a500fa",  # Dahlia
    "#00a1c2",  # Caribbean
    "#ec7000",  # Tangerine
    "#7645ff",  # Wisteria
    "#009f7a",  # Mediterranean
    "#ff4651",  # Blush
    "#108fd9",  # Sky
    "#c3404d",  # Rose
)

area_colors = NonSemanticColors(
    "#0085f2",  # Azure
    "#dd299d",  # Fuchsia
    "#69c300",  # Lime
    "#b429cc",  # Dahlia
    "#00bfd4",  # Caribbean
    "#ff9200",  # Tangerine
    "#6a4ce0",  # Wisteria
    "#19c295",  # Mediterranean
    "#ff6665",  # Blush
    "#24a4ee",  # Sky
    "#d94e6f",  # Rose
)

text_colors = NonSemanticColors(
    "#0059d3",  # Azure
    "#dd299d",  # Fuchsia
    "#007d00",  # Lime
    "#a500fa",  # Dahlia
    "#007e82",  # Caribbean
    "#ba5900",  # Tangerine
    "#6a4ce0",  # Wisteria
    "#008666",  # Mediterranean
    "#d63c45",  # Blush
    "#107bc2",  # Sky
    "#c3406f",  # Rose
)

semantic_text_colors = SemanticColors(
    "#7d14b6",  # Direct Effects: Violet
    "#4f53cc",  # Indirect Effects: Pale Blue Violet
    "#a30660",  # Specific Causes: Flamingo
    "#8f197b",  # General Causes: Carnation
    "#386a4d",  # Ongoing To Do: Dark Green
    "#476306",  # Immediate To Do: Pale Yellow Green
    "#0046e7",  # Administartion Ongoing: Blue
    "#0059d3",  # Administration Communications: Pale Blue
    "#386a4d",  # Administration Financial: Pale Green
    "#006860",  # Documentation Objective: Pale Aqua
    "#006860",  # Documentation Subjective: Aqua
    "#576375",  # Neutral: Gray
    "#0046e7",  # Neutral Themed: Themed (Blue in base theme)
    "#262e34",  # Do Not Use: Black
)

semantic_line_colors = SemanticColors(
    "#ab44ff",  # Direct Effects: Violet
    "#8c88ff",  # Indirect Effects: Pale Blue Violet
    "#d12ea1",  # Specific Causes: Flamingo
    "#9f3494",  # General Causes: Carnation
    "#4ea84e",  # Ongoing To Do: Dark Green
    "#abd456",  # Immediate To Do: Pale Yellow Green
    "#2588f1",  # Administartion Ongoing: Blue
    "#5ea1f8",  # Administration Communications: Pale Blue
    "#88c888",  # Administration Financial: Pale Green
    "#7ec0bb",  # Documentation Objective: Pale Aqua
    "#3ba59b",  # Documentation Subjective: Aqua
    "#a5b0b7",  # Neutral: Gray
    "#2588f1",  # Neutral Themed: Themed (Blue in base theme)
    "#4d5b69",  # Do Not Use: Black
)

io_area_colors = IOColors("#a2864b", "#f4c100")
io_line_colors = IOColors("#7a7000", "#bb9600")
io_text_colors = IOColors("#777242", "#8f7300")

# Light should be used for grid lines
# Medium should be used for axis lines/tick marks
# Dark for non-themed text
neutral_colors = NeutralColors("#e6e7e8", "#5f7c8a", "#3b434c")  # Light  # Medium  # Dark


def area_color_cycle() -> cycle:
    """Returns a cycle that goes through the area colors in sequence."""
    return cycle(area_colors)


def line_color_cycle() -> cycle:
    """Returns a cycle that goes through the line colors in sequence."""
    return cycle(line_colors)


def text_color_cycle() -> cycle:
    """Returns a cycle that goes through the text colors in sequence."""
    return cycle(text_colors)


color_string = "brgmcy"  # Single letter colors excluding black
# stash defaults to allow resets
defaults = {
    "letter_colors": {letter: mpl.colors.colorConverter.colors[letter] for letter in color_string},
    "axes.prop_cycle": mpl.rcParams["axes.prop_cycle"],
    "image.cmap": mpl.rcParams["image.cmap"],
}


def set_UX() -> None:
    """
    Primary function of the color module to set UX defined defaults for use in ``matplotlib.pyplot``.
    The colorcycler will use non-semantic line colors (NOTE: this includes bar charts).
    A discrete-colormap using non-semantic area colors is registered as 'UX' but still must be activated or specified.
    """
    mpl.style.use(_files(__package__) / "ux.mplstyle")

    # Register
    _register_colormaps()

    # we still need to set the color letters manually.
    set_color_letters()
    set_line_colors()
    set_discrete_colormap()


def _register_colormaps() -> None:
    """
    private method to setup color maps
    """
    area_cmap = LinearSegmentedColormap.from_list("Area", area_colors, N=len(area_colors))
    line_cmap = LinearSegmentedColormap.from_list("Line", line_colors, N=len(line_colors))
    text_cmap = LinearSegmentedColormap.from_list("Text", text_colors, N=len(text_colors))
    alert_cmap = LinearSegmentedColormap.from_list("Alert", alert_colors, N=len(alert_colors))
    neutral_cmap = LinearSegmentedColormap.from_list("Neutral", neutral_colors, N=len(neutral_colors))
    mpl.colormaps.register(cmap=area_cmap, force=True)
    mpl.colormaps.register(cmap=line_cmap, force=True)
    mpl.colormaps.register(cmap=text_cmap, force=True)
    mpl.colormaps.register(cmap=alert_cmap, force=True)
    mpl.colormaps.register(cmap=neutral_cmap, force=True)


def set_line_colors() -> None:
    """
    Sets line colors for ploting to match line colors.
    """
    mpl.rc("axes", prop_cycle=cycler("color", line_colors))


def reset_line_colors() -> None:
    """
    Unsets line colors, returning to the prior default.
    """
    mpl.rc("axes", prop_cycle=defaults["axes.prop_cycle"])


def set_discrete_colormap() -> None:
    """
    Sets the colormap for discrete colors to match area colors.
    """
    mpl.rc("image", cmap="Area")  # Set colormap separately as it is discrete; ie no gradients


def reset_discrete_colormap() -> None:
    """
    Unsets the colormap, returning to the prior default.
    """
    mpl.rc("image", cmap=defaults["image.cmap"])


def set_color_letters() -> None:
    """
    Sets color letters brgmcy to NonSemanticColors (b->Azure, r->Fuchsia, g->Lime, etc).
    """
    cc = mpl.colors.colorConverter
    for letter, hexColor in zip(color_string, line_colors[:7]):
        cc.colors[letter] = cc.to_rgb(hexColor)
        cc.cache[letter] = cc.to_rgb(hexColor)


def reset_color_letters() -> None:
    """
    Unsets color letters brgmcy to basic colors.
    """
    mpl.colors.colorConverter.colors.update(defaults["letter_colors"])
    mpl.colors.colorConverter.cache.update(defaults["letter_colors"])
