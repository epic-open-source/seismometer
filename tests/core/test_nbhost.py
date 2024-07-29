import os
from unittest.mock import patch

import pytest

import seismometer.core.nbhost as undertest


class TestNotebookHost:
    @pytest.mark.parametrize(
        "env_key,host",
        [
            ("VSCODE_CWD", undertest.NotebookHost.VSCODE),
            ("COLAB_JUPYTER_IP", undertest.NotebookHost.COLAB),
            ("JUPYTERHUB_HOST", undertest.NotebookHost.JUPYTER_HUB),
            ("JPY_PARENT_PID", undertest.NotebookHost.JUPYTER_LAB),
        ],
    )
    def test_current_host(self, env_key, host):
        with patch.dict(os.environ, {env_key: "1"}, clear=True):
            assert undertest.NotebookHost.get_current_host() == host

    @pytest.mark.parametrize(
        "env_key,supports_iframe",
        [
            ("VSCODE_CWD", False),
            ("COLAB_JUPYTER_IP", False),
            ("JUPYTERHUB_HOST", True),
            ("JPY_PARENT_PID", True),
        ],
    )
    def test_supports_iframe(self, env_key, supports_iframe):
        with patch.dict(os.environ, {env_key: "1"}, clear=True):
            assert undertest.NotebookHost.supports_iframe() == supports_iframe
