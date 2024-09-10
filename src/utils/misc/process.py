"""Tools for subprocesses

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""

import os
import shutil
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from subprocess import PIPE, Popen
from typing import Optional, Callable
from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired

from src.utils import hydra_utils

logger = hydra_utils.get_pylogger(__name__)


def check_exec(exec_path=None, env=None, default=None, error_suffix=None) -> str:
    """Check if an executable exists

    Returns:
        the path to the executable if exists
    """
    if exec_path is not None:
        exec_cmd = str(exec_path)
    elif env is not None:
        exec_cmd = os.environ.get(env, default)
        if exec_cmd is None:
            raise RuntimeError(
                f"Environment variable {env} not set, and no default exec provided.\n{error_suffix}"
            )
    else:
        assert default is not None, "No command info found, please check the code."
        exec_cmd = default

    if shutil.which(exec_cmd) is None:
        raise RuntimeError(
            f"Command `{exec_cmd}` not found. Check if one of the following is provided:\n   exec_path: {exec_path}\n   env {env}: {os.environ.get(env, None)}\n   default: {default}\n{error_suffix}"
        )
    return str(exec_cmd)


def subprocess_run(cmd, env=None, quiet=False, cwd=None, prefix=">>> "):
    """Run shell subprocess"""

    if isinstance(cmd, str):
        import shlex

        cmd = shlex.split(cmd)
    proc = Popen(cmd, stdout=PIPE, stderr=PIPE, env=env, cwd=cwd)
    out = ""

    for line in iter(proc.stdout.readline, b""):  # type:ignore
        out_line = line.decode("utf-8").rstrip()
        out += out_line
        if not quiet:
            print("{}{}".format(prefix, out_line), flush=True)
    stdout, stderr = proc.communicate()
    err = stderr.decode("utf-8").strip("\n")
    return out, err


def mp_with_timeout(
    iter,
    func,
    n_proc: int = 1,
    timeout: int = 60,
    chunksize: int = 1,
    print_every_iter: int = 1000,
    error_report_fn: Optional[Callable] = None,
    **kwargs
):
    """Use pebble for multiprocessing with timeout

    Args:
        iter (Iterable): an iterative of a single argument for func
        func (Callable): work function that takes single argument
        n_proc (int): number of processes in parallel
        timeout (float): number of seconds allowed for each process before timeout
        chunksize (int): size of each iteration chunks. Default 1
        print_every_iter (int): print number of success every this interation
        error_report_fn (Callable): function from the index of the iterable to any output error message

    Returns:
        list of results. 'timeout' if the process timed out, 'error' if the process expired.
    """
    if error_report_fn is None:

        def error_report_fn(ix):
            return ix
    if len(kwargs) > 0:
        func = partial(func, **kwargs)

    all_results = []
    with ProcessPool(n_proc) as p:
        future = p.map(func, iter, chunksize=chunksize, timeout=timeout * chunksize)
        result_itr = future.result()

        with tqdm(total=len(iter)) as pbar:
            current_ix = 0
            success_ix = 0
            intv_counter = 0
            while True:
                try:
                    result = next(result_itr)
                    all_results.append(result)
                    success_ix += 1
                except StopIteration:
                    break
                except TimeoutError as error:
                    logger.error(
                        f"Timeout {error.args[1]} s: {error_report_fn(current_ix)}"
                    )
                    all_results.append("timeout")
                except ProcessExpired as error:
                    logger.error(
                        f"{error} (exit code: {error.exitcode}): {error_report_fn(current_ix)}"
                    )
                    all_results.append("error")
                pbar.update(1)
                current_ix += 1
                intv_counter += 1
                if print_every_iter and intv_counter == print_every_iter:
                    logger.info(f"{current_ix}/{len(iter)}: success: {success_ix}")
                    intv_counter = 0

    return all_results


def mp_imap_unordered(
    iter, func, n_proc: int = 1, chunksize: int = 1, mute_tqdm: bool = False, **kwargs
):
    """Helper function for multiprocessing run. The each item in the iterable should contain only one argument"""

    if len(kwargs) > 0:
        func = partial(func, **kwargs)

    if n_proc > 1:
        with mp.Pool(n_proc) as p:
            try:
                results = list(
                    tqdm(
                        p.imap_unordered(func, iter, chunksize=chunksize),
                        total=len(iter),
                        disable=mute_tqdm,
                    )
                )
                p.close()
                p.terminate()
            except KeyboardInterrupt:
                print("Received KeyboardInterrupt, terminating processes..", flush=True)
                p.terminate()
                p.join()
                results = []
    else:
        results = [func(x) for x in tqdm(iter, disable=mute_tqdm)]

    return results


def mp_imap(
    iter, func, n_proc: int = 1, chunksize: int = 1, mute_tqdm: bool = False, **kwargs
):
    """Helper function for multiprocessing run. The each item in the iterable should contain only one argument"""

    if len(kwargs) > 0:
        func = partial(func, **kwargs)

    if n_proc > 1:
        with mp.Pool(n_proc) as p:
            try:
                results = list(
                    tqdm(
                        p.imap(func, iter, chunksize=chunksize),
                        total=len(iter),
                        disable=mute_tqdm,
                    )
                )
                p.close()
                p.terminate()
            except KeyboardInterrupt:
                print("Received KeyboardInterrupt, terminating processes..", flush=True)
                p.terminate()
                p.join()
                results = []
    else:
        results = [func(x) for x in tqdm(iter, disable=mute_tqdm)]

    return results
