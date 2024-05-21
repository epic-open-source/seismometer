import logging
import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from typing import Optional

from seismometer import __version__ as version

from .compile import build_parser, compile_notebook
from .extract import extract_parser, extract_supplement

CMDROUTER = {
    "build": compile_notebook,
    "extract": extract_supplement,
}


def create_parser() -> ArgumentParser:
    """Create the parser for the seismometer CLI."""
    parser = ArgumentParser(
        description="seismometer"
        + "\n  Provides commands for building model-specific notebooks after"
        + "\n  extracting and modifying supplemental information from a template.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {version}")

    subparsers = parser.add_subparsers(dest="subcommand", metavar="COMMAND")

    sub_builder = subparsers.add_parser(
        "build",
        help="Build a seismograph notebook",
        description="Build the seismograph notebook from a template and supplement.",
    )
    build_parser(sub_builder)

    sub_extract = subparsers.add_parser(
        "extract",
        help="Extract the seismograph prose",
        description="Extract config from a seismometer.",
    )
    extract_parser(sub_extract)

    return parser


def main_cli(args: Optional[list[str]] = None) -> None:
    """
    The primary entrypoint for the seismometer CLI.

    This both creates the parser then routes to the subcommand.
    Two special flags are handled here:

    - `--pdb` to drop into the debugger on an exception,
    - `--verbose` changes the log level to logging.INFO.

    These are not passed to the subcommand.

    Parameters
    ----------
    args : Optional[list[str]], optional
        List of arguments to parse, by default None; uses commandline arguments.
    """
    parser = create_parser()

    # Action
    kwargs = vars(parser.parse_args(args))
    use_pdb = kwargs.pop("pdb", False)
    if kwargs.pop("verbose", False):
        logging.basicConfig(level=logging.INFO)
    cmd = kwargs.pop("subcommand", None)  # default to only subcommand?
    if cmd is None:
        parser.print_help()
        sys.exit(1)
    try:
        CMDROUTER[cmd](**kwargs)
        sys.exit(0)
    except Exception as exc:  # pragma: no cover
        if not use_pdb:
            print(str(exc))
            print(f"Use seismometer {cmd} --help for more information.")
            sys.exit(1)

        import pdb
        import traceback

        print(f">> Error:: {traceback.format_exception_only(type(exc), exc)[0].strip()}")
        pdb.post_mortem(exc.__traceback__)


if __name__ == "__main__":
    main_cli()
