import os
from enum import Enum


class NotebookHost(Enum):
    JUPYTER_LAB = 1
    JUPYTER_HUB = 2
    VSCODE = 3
    COLAB = 4
    OTHER = -1

    @classmethod
    def get_current_host(cls):
        env_keys = list(os.environ.keys())

        if "VSCODE_CWD" in env_keys:
            return cls.VSCODE
        if "COLAB_JUPYTER_IP" in env_keys:
            return cls.COLAB
        elif "JUPYTERHUB_HOST" in env_keys:
            return cls.JUPYTER_HUB
        elif "JPY_PARENT_PID" in env_keys:
            return cls.JUPYTER_LAB
        return cls.OTHER

    @classmethod
    def supports_iframe(cls) -> bool:
        """
        If IFrames can be used to host local content.

        Colab and VScode do not support IFrames.
        """
        return cls.get_current_host() in [cls.JUPYTER_LAB, cls.JUPYTER_HUB]
