from enum import Enum

import psutil


class NotebookHost(Enum):
    JUPYTER_LAB = 1
    JUPYTER_HUB = 2
    VSCODE = 3
    OTHER = -1

    @classmethod
    def get_current_host(cls):
        cmds = [x for x in psutil.Process().parent().cmdline()]

        if any("jupyter-lab" in cmd for cmd in cmds):
            return cls.JUPYTER_LAB
        elif any("jupyterhub" in cmd for cmd in cmds):
            return cls.JUPYTER_HUB
        elif any("vscode-server" in cmd for cmd in cmds):
            return cls.VSCODE
        return cls.OTHER
