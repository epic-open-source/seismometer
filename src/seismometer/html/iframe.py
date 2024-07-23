from pathlib import Path

from IPython.display import IFrame


def load_as_iframe(path: Path, height: int = 300) -> IFrame:
    return IFrame(src=path.relative_to(Path.cwd()), width="100%", height=height)
