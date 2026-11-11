"""SIFT1M (INRIA TEXMEX corpus): 1M base and 100k learn SIFT descriptors of dimension 128."""

import tarfile
import urllib.request
from pathlib import Path

import numpy as np

SIFT_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"
SIFT_FILES = ("sift/sift_base.fvecs", "sift/sift_learn.fvecs")


def read_fvecs(path: Path) -> np.ndarray:
    """Read an .fvecs file: each vector is an int32 dimension followed by that many float32."""
    raw = np.fromfile(path, dtype="int32")
    d = raw[0]
    return np.ascontiguousarray(raw.reshape(-1, d + 1)[:, 1:].view("float32"))


def download_sift1m(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "sift.tar.gz"
    print(f"Downloading {SIFT_URL} (168 MB)...")
    urllib.request.urlretrieve(SIFT_URL, archive)
    with tarfile.open(archive) as tar:
        tar.extractall(data_dir, members=[tar.getmember(name) for name in SIFT_FILES], filter="data")
    archive.unlink()


def load_sift1m(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the 1M base vectors (training set) and the 100k learn vectors (held-out set)."""
    if not all((data_dir / name).exists() for name in SIFT_FILES):
        download_sift1m(data_dir)
    base, learn = (read_fvecs(data_dir / name) for name in SIFT_FILES)
    return base, learn
