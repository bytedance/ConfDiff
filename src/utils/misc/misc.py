"""Misc tools

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""

# =============================================================================
# Imports
# =============================================================================
import shutil
import tempfile
import subprocess
import pickle
from pathlib import Path
from torch import Tensor


# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Functions
# =============================================================================

def load_pickle(fpath):
    with open(fpath, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_pickle(obj, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f)


def get_git_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


def get_persist_tmp_fpath(suffix='.pdb'):
    """Get a temp file path. The file is not deleted until the end of the session."""
    return tempfile.NamedTemporaryFile(suffix=suffix, delete=False).name


def unbatch_dict(d):
    """unbatch the lenght 1 dict object due to PyG batching"""
    if isinstance(d, dict):
        return {key: unbatch_dict(val) for key, val in d.items()}
    elif isinstance(d, list):
        return d[0]
    elif isinstance(d, Tensor):
        if d.shape == (1,):
            # single value Tensor
            return d[0].cpu().item()
        else:
            # other tensor
            return d[0]
    else:
        raise TypeError(f"Unknown type: {d}")


def gather_all_files(file_root, remove_subfolder=False):
    """Gather all files in sub-folders under the file_root"""
    file_root = Path(file_root)
    all_files = [f for f in file_root.rglob("*.*") if f.is_file()]
    for f in all_files:
        if not file_root.joinpath(f.name).exists():
            shutil.move(str(f), str(file_root))
    if remove_subfolder:
        all_dirs = [f for f in file_root.glob("*") if f.is_dir()]
        for d in all_dirs:
            shutil.rmtree(str(d))


def flatten_list(lst):
    """Flatten a list recursively"""
    if isinstance(lst, (list, tuple)):
        f_l = []
        for a in lst:
            f_l += flatten_list(a)
        return f_l
    else:
        return [lst]


def replace_dot_key_val(d, dot_key, replace_to, inplace=True, ignore_error=False):
    """Replace the value in an hierachical dict with dot format key"""
    if not inplace:
        from copy import deepcopy
        d = deepcopy(d)
    key_levels = dot_key.split('.')
    node = d
    try:
        if len(key_levels) > 1:
            for key in key_levels[:-1]:
                node = node[key]
        assert key_levels[-1] in node.keys(), f"{dot_key} not found"
        node[key_levels[-1]] = replace_to
    except Exception as e:
        if ignore_error:
            pass
        else:
            raise e
    return d

# =============================================================================
# Classes
# =============================================================================
