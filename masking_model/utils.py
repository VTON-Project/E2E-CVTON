from pathlib import Path

def convert_to_path(*args) -> list[Path]:
    paths = []
    for p in args:
        if p is None: paths.append(None)
        else: paths.append(Path(p))
        
    return paths