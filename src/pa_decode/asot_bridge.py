"""Import the authors' ASOT code (``mingu6/action_seg_ot``) pinned at commit ``0c4c86b1``.

The repository is expected at ``third_party/action_seg_ot`` (run ``bash setup.sh``) or at the path given by the
environment variable ``ASOT_ROOT``.
"""
import os
import subprocess
import sys

import numpy as np

from .config import ASOT_COMMIT

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASOT_ROOT = os.environ.get('ASOT_ROOT', os.path.join(REPO_ROOT, 'third_party', 'action_seg_ot'))

if not os.path.isdir(os.path.join(ASOT_ROOT, 'src')):
    raise ImportError(f'ASOT code not found at {ASOT_ROOT}. Run `bash setup.sh` or set ASOT_ROOT.')


def asot_commit():
    try:
        return subprocess.check_output(['git', '-C', ASOT_ROOT, 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return 'unknown'


ASOT_HEAD = asot_commit()
if ASOT_HEAD != 'unknown' and not ASOT_HEAD.startswith(ASOT_COMMIT):
    raise ImportError(f'ASOT must be at commit {ASOT_COMMIT}, found {ASOT_HEAD}. Run `bash setup.sh`.')

if os.path.join(ASOT_ROOT, 'src') not in sys.path:
    sys.path.insert(0, os.path.join(ASOT_ROOT, 'src'))

# numpy >= 2 removed random_integers, which the authors' code uses; random_integers(lo, hi, n) == randint(lo, hi + 1, n)
if not hasattr(np.random, 'random_integers'):
    np.random.random_integers = lambda low, high=None, size=None: np.random.randint(low, high + 1, size)

os.environ.setdefault('WANDB_MODE', 'disabled')

import asot  # noqa: E402,F401


def load_training_modules():
    """Modules that need pytorch-lightning / wandb (only for training and evaluation of the unsupervised pipeline)."""
    import metrics as amet
    import train as atrain
    from video_dataset import VideoDataset
    return amet, atrain, VideoDataset
