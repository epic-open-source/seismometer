import logging
from typing import Any

from IPython.display import HTML, SVG
from jinja2 import Environment, PackageLoader, TemplateNotFound

logger = logging.getLogger("seismometer")

# Initializing Jinja
package_loader = PackageLoader("seismometer", "html/resources")
jinja2_env = Environment(lstrip_blocks=True, trim_blocks=True, loader=package_loader)


def render_info_template(info_vals: dict[str, Any]) -> HTML:
    """
    Get templated HTML containing information about the available datasets.

    Parameters
    ----------
    info_vals : dct[str, Any]
        Dictionary of values required by the info template.

    Returns
    -------
    HTML
        The templated HTML object.
    """
    return render_into_template("info", info_vals)


def render_cohort_summary_template(dfs: dict[str, list[str]]) -> HTML:
    """
    Get templated HTML containing the cohort summaries.

    Parameters
    ----------
    dfs : dict[str, list[str]]
        Dictionary of HTML-ified tables by cohort.

    Returns
    -------
    HTML
        The templated HTML.
    """
    return render_into_template("cohorts", {"dfs": dfs})


def render_into_template(name: str, values: dict = None) -> HTML:
    """
    Uses jinja to render a dictionary of values into a template.

    Parameters
    ----------
    name : str
        The template name.
    values : Optional[dict], optional
        A dictionary of values to be templated into the HTML, by default None.

    Returns
    -------
    HTML
        The templated HTML
    """
    values = values or {}
    try:
        template = jinja2_env.get_template(f"{name}.html")
    except TemplateNotFound:
        logger.warning(f"HTML template {name} not found.")
        return HTML()

    return HTML(template.render(values))


def render_title_message(title: str, message: str) -> HTML:
    """
    Simple title + message template

    Parameters
    ----------
    title : str
        The title for the section.
    message : str
        The message to display.

    Returns
    -------
    HTML
        The templated HTML.
    """
    return render_into_template("title_message", {"title": title, "message": message})


def render_censored_plot_message(censor_threshold: int) -> HTML:
    """
    Get templated HTML containing a censored plot.

    Parameters
    ----------
    censor_threshold : int
        Required threshold for plotting.

    Returns
    -------
    HTML
        The templated HTML.
    """
    return render_title_message("Data is censored.", f"There are {censor_threshold} or fewer rows.")


def render_title_with_image(title: str, image: SVG) -> HTML:
    """
    Simple title + image template

    Parameters
    ----------
    title : str
        The title for the section.
    message : str
        The message to display.

    Returns
    -------
    HTML
        The templated HTML.
    """
    return render_into_template("title_image", {"title": title, "image": image.data})
