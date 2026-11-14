"""Download the reference datasets once so the test suite runs offline.

    python tests/data/fetch_datasets.py
"""

import urllib.request
from pathlib import Path

from sklearn.datasets import fetch_california_housing, load_iris

HERE = Path(__file__).parent
TITANIC_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def iris() -> None:
    frame = load_iris(as_frame=True).frame
    frame["species"] = frame.pop("target").map(dict(enumerate(load_iris().target_names)))
    frame.to_csv(HERE / "iris.csv", index=False)


def california() -> None:
    frame = fetch_california_housing(as_frame=True).frame
    frame.to_csv(HERE / "california_housing.csv", index=False)


def titanic() -> None:
    with urllib.request.urlopen(TITANIC_URL, timeout=60) as resp:
        (HERE / "titanic.csv").write_bytes(resp.read())


if __name__ == "__main__":
    for fetch in (iris, california, titanic):
        fetch()
        print("ok", fetch.__name__)
