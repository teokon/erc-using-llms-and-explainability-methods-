"""Central, portable path + device configuration for the whole repository.

Every training / explainability / plotting script imports its directories from here instead of
hard-coding machine-specific absolute paths, so the code runs unchanged on a laptop, a cluster, or
a Code Ocean capsule.

All locations are resolved relative to this file (the repository root) and can be overridden with
environment variables — this is exactly what the Code Ocean capsule does (it points the inputs at
``/data`` and the outputs at ``/results``):

    ERC_DATA          dataset CSVs             (default: <repo>/Datasets)
    ERC_CHECKPOINTS   trained models + saved   (default: <repo>/checkpoints)
                      attributions/embeddings
    ERC_RESULTS       generated figures/tables (default: <repo>/results)
    OPTIMUS_REPO      vendored Optimus library (default: <repo>/third_party/optimus)

GPU selection: ``pick_gpu()`` pins CUDA to a single device with the precedence
``$GPU  >  an already-single-value CUDA_VISIBLE_DEVICES  >  "0"``. Call it *before* importing torch.

Typical use from a script inside Models/ or Explainability/::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from erc_paths import DATA_DIR, CHECKPOINTS_DIR, RESULTS_DIR, OPTIMUS_REPO, pick_gpu
"""
import os
from pathlib import Path

# Repository root = the directory that contains this file.
REPO_ROOT = Path(__file__).resolve().parent


def _dir(env_var: str, default_rel: str) -> Path:
    """Resolve a directory from an env var, falling back to a repo-relative default."""
    val = os.environ.get(env_var)
    return Path(val).expanduser().resolve() if val else (REPO_ROOT / default_rel)


# Input datasets (raw + context-constructed CSVs).
DATA_DIR = _dir("ERC_DATA", "Datasets")
# Trained checkpoints and the saved attributions / embeddings the figure scripts read.
CHECKPOINTS_DIR = _dir("ERC_CHECKPOINTS", "checkpoints")
# Where generated figures / tables are written.
RESULTS_DIR = _dir("ERC_RESULTS", "results")
# Vendored patched Optimus library (see third_party/optimus/PATCHES.md).
OPTIMUS_REPO = _dir("OPTIMUS_REPO", "third_party/optimus")


def pick_gpu() -> str:
    """Pin CUDA to ONE device and return the chosen id (as a string).

    The shell often presets ``CUDA_VISIBLE_DEVICES="0,1,2,3"``; left as-is that makes every run land
    on GPU 0. Precedence: ``$GPU`` > an already-single-value ``CUDA_VISIBLE_DEVICES`` > ``"0"``.
    Must be called before torch is imported.
    """
    gpu = os.environ.get("GPU")
    if not gpu:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        gpu = cvd if (cvd and "," not in cvd) else "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return gpu
