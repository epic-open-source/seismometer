import logging
from argparse import ArgumentParser
from pathlib import Path

from seismometer.configuration import ConfigProvider, Option, template_options
from seismometer.core.io import load_markdown, load_notebook, write_json, write_notebook

from .arguments import common_builder_arguments
from .jupyter import contrib_cells, get_id

# region CLI


@common_builder_arguments
def build_parser(parser: ArgumentParser) -> ArgumentParser:
    """Create the subcommand parsers for building a notebook."""
    parser.add_argument(
        "-i",
        "--input_dir",
        dest="supplement",
        type=Path,
        help="Path to the supplemental markdown; overrides the values in the config.yml.",
    )
    parser.add_argument(
        "--md",
        "--markdown-to-json",
        dest="md_only",
        action="store_true",
        help="Debugging function to compile the markdown files into a single json.",
    )


# endregion


def compile_notebook(
    config_yaml: Path,
    *,
    template: Option,
    supplement: Path = None,
    output: Path = None,
    template_file: Path = None,
    md_only: bool = False,
    force: bool = False,
) -> None:
    """
    Function to compile a notebook.

    Compile a notebook from the specified inputs.
    It is expected that most usage only passes in the config_yaml path
    (YAML file containing "other_info" top-level key).

    Furthermore, config_yaml parameter defaults to "./config.yml" allowing for no explicit arguments.
    In these cases, the other_info must specify locations for all other information.

    Parameters
    ----------
    config_yaml : Path
        Path to the configuration yaml file, must contain an "other_info" key, by default "./config.yml".
    template : Option
        Name of the template to use. This parameter is ignored if template_file is specified. It takes precedence over
        the template specified in configuration files.
    supplement : Path, optional
        Location of the markdown files to compile into the template. If not specified, info_dir must be included in
        the configuration files.
    output : Optional[Path], optional
        Location to write the output notebook, by default None; it uses the info_dir where supplement information is
        located.
    template_file : Optional[Path], optional
        Location of a custom template. It takes precedence over the template argument and configuration,
        by default None.
    md_only : bool, optional
        If True, it will not write the compiled notebook and instead compiles all markdown into a single JSON for
        review, by default False.

        The compiled JSON is written to the supplement folder and is NOT used for compilation of notebooks. It is
        created only to simplify review.
    force : bool, optional
        If True, overwrites any existing files, by default False.
    """
    config = ConfigProvider(config_yaml, template_notebook=template, info_dir=supplement)
    config.set_output(output, nb_prefix="gen_")

    # Three options for specifying template
    if template_file is not None:
        template = template_options.add_adhoc_template(template_file)
    elif template is not None:
        template = template_options[template]
    else:
        template = config.template

    supplement_dict = _load_supplement(config.info_dir)
    if md_only:
        _combine_markdown(config, supplement_dict, force)

    notebook = load_notebook(nb_template=template)

    for cell in contrib_cells(notebook):
        _add_markdown(cell, supplement_dict)

    write_notebook(notebook, config.output_dir / config.output_notebook, overwrite=force)


def _add_markdown(cell, content: dict) -> None:
    """Inserts a markdown into a single cell."""
    key = get_id(cell)
    try:
        cell["source"] = content[key]
    except KeyError:
        logging.warning(f"Markdown configuration needs content for key {key}.")
        return


def _load_supplement(res_dir: Path) -> dict:
    """Load all markdown (*.md) files found in the directory."""
    supplement = {}
    for file in Path(res_dir).glob("*.md"):
        supplement[file.stem] = "".join(load_markdown(file))
    return supplement


def _combine_markdown(config: "Configuration", supplement_dict: dict, overwrite: bool):
    """Merges all read markdowns into a single JSON file."""
    write_json(supplement_dict, config.output_dir / "supplement.json", overwrite=overwrite)
