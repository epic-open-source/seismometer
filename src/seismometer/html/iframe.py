from pathlib import Path

from IPython.display import HTML, IFrame

from ..core.nbhost import NotebookHost


def load_as_iframe(path: Path, height: int = 300) -> IFrame | HTML:
    if NotebookHost.supports_iframe():
        return IFrame(src=path.relative_to(Path.cwd()), width="100%", height=height)
    return HTML(data=f'<div style="width:100%;height:{height}px;p>{path.read_text()}</div>')
