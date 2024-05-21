from argparse import ArgumentParser
from pathlib import Path
from unittest.mock import Mock, call, patch

import pytest

import seismometer.builder as undertest


class TestRoutingToBuild:
    def test_no_args_sets_defaults(self):
        expected_call = call(
            config_yaml=Path("config.yml"),
            supplement=None,
            md_only=False,
            template=None,
            template_file=None,
            output=None,
            force=False,
        )
        spy_cmd = Mock()
        with patch.dict(undertest.CMDROUTER, {"build": spy_cmd}):
            with pytest.raises(SystemExit) as sysexit:
                undertest.main_cli(["build"])

        assert sysexit.value.code == 0
        assert spy_cmd.call_args == expected_call


class TestcompileParser:
    def test_defaults(self):
        parser = ArgumentParser()
        undertest.build_parser(parser)

        args = parser.parse_args([])

        assert args.config_yaml == Path("config.yml")
        assert args.template is None
        assert args.template_file is None
        assert args.output is None
        assert args.force is False

    @pytest.mark.parametrize(
        "input,expected_arg,expected_val",
        [
            ("-c test.txt", "config_yaml", "test.txt"),
            ("--config_yaml test.txt", "config_yaml", "test.txt"),
            ("--file subdir/ondem.json", "template_file", "subdir/ondem.json"),
            ("--output adir/example.ipynb", "output", "adir/example.ipynb"),
            ("-i subdir/ondem.json", "supplement", "subdir/ondem.json"),
        ],
    )
    def test_path_args(self, input, expected_arg, expected_val):
        input_path = Path(expected_val)
        parser = ArgumentParser()
        undertest.build_parser(parser)
        parsed = parser.parse_args(input.split(" "))

        assert vars(parsed).get(expected_arg) == input_path

    @pytest.mark.parametrize(
        "input,expected_arg,expected_val",
        [
            ("--force", "force", True),
            ("-t binary", "template", "binary"),
            ("--template binary", "template", "binary"),
            ("--markdown-to-json", "md_only", True),
            ("--md", "md_only", True),
        ],
    )
    def test_add_args(self, input, expected_arg, expected_val):
        parser = ArgumentParser()
        undertest.build_parser(parser)
        parsed = parser.parse_args(input.split(" "))

        assert vars(parsed).get(expected_arg) == expected_val
