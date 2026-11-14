import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 40)


def sh(cmd):
    print('$', cmd)
    subprocess.run(cmd, shell=True, check=True)


def pip_install(*pkgs):
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs], check=True)


def show(df, title=None, digits=2):
    if title:
        print(title)
    print(df.round(digits).to_string() if hasattr(df, 'round') else df)
    print()


def t_ci(x):
    """95% Student-t confidence interval of the mean."""
    x = np.asarray(x, float)
    if len(x) < 2:
        return (np.nan, np.nan)
    h = student_t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return (x.mean() - h, x.mean() + h)


def _default(o):
    if hasattr(o, 'tolist'):
        return o.tolist()
    if hasattr(o, 'item'):
        return o.item()
    return str(o)


def save_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=_default)
    print('saved', path)


def key_str(k):
    return '|'.join(map(str, k))
