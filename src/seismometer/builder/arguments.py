from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

from seismometer.configuration import template_options

# region Shared Arguments


def common_builder_arguments(create_subcommand_parser: Callable[[ArgumentParser], ArgumentParser]):
    def wrapped_create_parser(parser: ArgumentParser):
        _add_config_arg(parser)
        create_subcommand_parser(parser)
        _add_template_arg(parser)
        _add_templatefile_arg(parser)
        _add_output_arg(parser)
        _add_overwrite_arg(parser)
        parser.add_argument("--pdb", action="store_true", help="Drop into pdb on error.")
        parser.add_argument("--verbose", action="store_true", help="Print additional error messages.")

    return wrapped_create_parser


def _add_config_arg(parser: ArgumentParser, default=None) -> None:
    parser.add_argument(
        "-c",
        "--config_yaml",
        type=Path,
        default="config.yml",
        help="Specify the base configuration yaml file. \
                            This may contain other config options or references to additional configuration.",
    )


def _add_output_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="File or directory location to write the output, defaults to current working directory.",
    )


def _add_template_arg(parser: ArgumentParser, default=None) -> None:
    parser.add_argument(
        "-t",
        "--template",
        type=lambda opt: template_options[opt].name,
        choices=list(template_options),
        default=default,
        help="Choose the template to start with. This is usually based on the type of prediction the model makes.",
    )


def _add_templatefile_arg(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--file",
        dest="template_file",
        type=Path,
        help="Path to a custom template notebook to extract from; when present ignores --template.",
    )


def _add_overwrite_arg(parser: ArgumentParser) -> None:
    parser.add_argument("--force", action="store_true", help="Overwrite the output files if they exists.")


# endregion
