import shutil
from pathlib import Path

import polars as pl


def make_dir_if_not_exists(path):
    p = Path(path)
    target = p if p.suffix == '' else p.parent
    target.mkdir(parents=True, exist_ok=True)


def clear_dir(dir_path):
    p = Path(dir_path)
    if not p.exists():
        return
    for item in p.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def remove_dir(dir_path):
    p = Path(dir_path)
    if not p.exists():
        return
    shutil.rmtree(p)


def find_project_root(start=None):
    start = start or Path(__file__).resolve()

    for p in [start, *start.parents]:
        if (p / 'pyproject.toml').exists() or (p / '.git').exists():
            return p

    raise RuntimeError('Project root not found.')


def set_polars_visual_dims(rows=None, cols=None):
    if rows:
        pl.Config.set_tbl_rows(rows)
    if cols:
        pl.Config.set_tbl_cols(cols)
