import argparse
from unittest.mock import patch

import pytest

import seismometer.builder as undertest


def fake_cmd(*args, **kwargs):
    pass


class TestCLI:
    def test_bad_command_exits_2(self, capsys):
        with pytest.raises(SystemExit) as sysexit:
            undertest.main_cli(["notacommand"])
        assert sysexit.value.code == 2

    def test_no_command_exits_1(self, capsys):
        with pytest.raises(SystemExit) as sysexit:
            undertest.main_cli([])
        assert sysexit.value.code == 1

    @patch.dict(undertest.CMDROUTER, {"build": fake_cmd})
    def test_real_command_exits_0(self):
        with pytest.raises(SystemExit) as sysexit:
            undertest.main_cli(["build"])
        assert sysexit.value.code == 0


class TestCreateParser:
    def test_version_argument(self):
        parser = undertest.create_parser()

        version_action = parser._actions[1]
        assert isinstance(version_action, argparse._VersionAction)
        assert version_action.option_strings == ["-v", "--version"]
        assert version_action.version == f"%(prog)s {undertest.version}"

    def test_supported_subcommands(self):
        parser = undertest.create_parser()
        subparsers = parser._actions[2]
        assert "build" in subparsers.choices
        assert "extract" in subparsers.choices
