from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from seismometer.configuration import ConfigProvider, Option, template_options
from seismometer.core.io import load_notebook, write_markdown

from .arguments import common_builder_arguments
from .jupyter import contrib_cells, get_id, get_text

# region CLI


@common_builder_arguments
def extract_parser(parser: ArgumentParser) -> ArgumentParser:
    """Create the subcommand parsers for extracting a notebook."""
    pass  # Only contains common arguments


# endregion


def extract_supplement(
    config_yaml: Path, *, output: Path = None, template: Option, template_file: Path = None, force: bool = False
) -> None:
    """
    Function to extract the base supplement from a notebook.

    Extracts all contribution cells from a template notebook and writes them as markdown files.

    Parameters
    ----------
    config_yaml : Path
        Path to the configuration yaml file, must contain an "other_info" key.
    output : Optional[Path], optional
        Location to write the output files, by default it uses the info_path where supplement information is located.
    template : Option
        Name of the template to use. This parameter is ignored if template_file is specified. It takes precedence over
         the template specified in configuration files.
    template_file : Optional[Path], optional
        Location of a custom template. It takes precedence over the template argument and configuration,
        by default None.
    force : bool, optional
        If True, overwrites any existing files, by default False.
    """
    config = ConfigProvider(config_yaml, template_notebook=template)
    config.set_output(output)

    # Three options for specifying template
    if template_file is not None:
        template = template_options.add_adhoc_template(template_file)
    elif template is not None:
        template = template_options[template]
    else:
        template = config.template

    notebook = load_notebook(nb_template=config.template)
    notemap = _extract_notemap(notebook)

    _dump_markdown(notemap, config.output_dir, overwrite=force)


def _extract_notemap(notebook: Any, from_template: bool = False):
    """Extracts the contribution cells (tag of sg-contrib) from a notebook."""
    notemap = {}
    for cell in contrib_cells(notebook):
        cell_id = get_id(cell)
        notemap[cell_id] = get_text(cell, from_template)

    return notemap


def _dump_markdown(notemap: dict, output_dir: Path, overwrite: bool = False):
    """Writes the notemap to markdown files based on the key sg-id."""
    for key, value in notemap.items():
        write_markdown(value, output_dir / f"{key}.md", overwrite=overwrite)


def generate_data_dict_from_parquet(inpath: Path | str, outpath: Path | str, section: str = "predictions"):
    """
    Generate a data dictionary YAML file from a Parquet file.

    Parameters
    ----------
    inpath : Path | str
        The path to the input Parquet file.
    outpath : Path | str
        The path to the output YAML file.
    section : str, optional
        The section name to be used in the YAML file, by default "predictions".
    """
    import pandas as pd
    import yaml

    df = pd.read_parquet(inpath)

    items = []

    for c in df.columns:
        items.append({"name": c, "dtype": str(df[c].dtype), "definition": f"Placeholder description for {c}"})

    with open(outpath, "w") as f:
        f.write(yaml.dump({section: items}))
