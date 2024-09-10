"""Methods to evaluate results for MD sampled distributions

----------------
Copyright (2024) Bytedance Ltd. and/or its affiliates
SPDX-License-Identifier: Apache-2.0
"""

# =============================================================================
# Imports
# =============================================================================
from typing import Tuple
import warnings
warnings.filterwarnings("ignore", message=".*invalid value encountered in log10.*")
warnings.filterwarnings("ignore", message=".*divide by zero encountered in log.*")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.linalg import sqrtm
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon

from src.utils.misc.process import mp_imap
from .mdtraj_tools import filter_CA_pairwise_dist

# =============================================================================
# Constants
# =============================================================================

# =============================================================================
# Functions
# =============================================================================


def compute_contact_rate(pdist, pair_idx, seqlen, cutoff_nm=1.):
    """Compute the contact rate matrix"""
    contact_pct = pd.DataFrame(pair_idx, columns=['res_1', 'res_2'])
    contact_pct['contact_rate'] = np.mean(pdist < cutoff_nm, axis=0)
    res_list = np.arange(seqlen)
    contact_rate = contact_pct.pivot(index='res_1', columns='res_2', values='contact_rate').reindex(index=res_list, columns=res_list).fillna(0)
    contact_rate += contact_rate.values.T
    contact_rate += np.eye(seqlen)
    return contact_rate.values


def compute_hist_counts_2d(data, bins):
    """Compute counts of 2D data in each bins"""

    assert len(data.shape) == 2, "data should have shape: (n_sample, n_feat)"
    if data.shape[1] > 2:
        print(f"{data.shape[1]} features found, will only use first two (0, 1) features for plotting")

    H, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], bins=bins, density=False)

    # H has shape (xs, ys) -> (ys, xs) for visualization
    H = H.transpose()

    if isinstance(bins, int):
        xbins = ybins = bins
    else:
        xbins, ybins = bins
    xs = (xedges[:-1] + xedges[1:]) / 2
    ys = (yedges[:-1] + yedges[1:]) / 2
    xs = np.tile(np.expand_dims(xs, 0), (ybins, 1))
    ys = np.tile(np.expand_dims(ys, 1), (1, xbins))
    # print(H.shape, xs.shape, ys.shape)
    return H, xs, ys, xedges, yedges


def hist_contour_plot(
    # args to compute the histogram
    data=None, bins=50, 
    H=None, xedges=None, yedges=None,
    # args for plotting
    ax=None, logscale=True, levels=10, linewidths=1, cmap='GnBu', return_all=False, **kwargs
):
    if data is None:
        assert H is not None, "H is required if data is None"
        assert xedges is not None, "xedges is required if data is None"
        assert yedges is not None, "yedges is required if data is None"
        ybins, xbins = H.shape   # NOTE: H shape should be already transposed
        xs = (xedges[:-1] + xedges[1:]) / 2
        ys = (yedges[:-1] + yedges[1:]) / 2
        xs = np.tile(np.expand_dims(xs, 0), (ybins, 1))
        ys = np.tile(np.expand_dims(ys, 1), (1, xbins))
    else:
        H, xs, ys, xedges, yedges = compute_hist_counts_2d(data, bins=bins)

    if logscale:
        # log10: density -> energy
        H_show = H.copy()
        H_show[np.where(H_show == 0)] = -np.inf
        H_show = np.log10(H_show, where=~np.isnan(H_show))
    else:
        H_show = H
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.contour(xs, ys, H_show, levels=levels, linewidths=linewidths, cmap=cmap, **kwargs)
    ax.set_xlabel('tIC 1')
    ax.set_ylabel('tIC 2')

    if return_all:
        return ax, H, xedges, yedges
    else:
        return ax


def scatterplot_with_density(data, max_kde_sample=None, max_sample_show=None, scatter_sample_order=0.5, ax=None, **kwargs):
    """Scatterplot with density estimation"""

    # compute KDE
    if max_kde_sample is not None and data.shape[0] > max_kde_sample:
        idx = np.random.choice(np.arange(data.shape[0]), size=max_kde_sample, replace=False)
        kde_sample = data[idx, ...]
    else:
        kde_sample = data
    kde = gaussian_kde(kde_sample.T)

    # scatter plots
    if max_sample_show is not None and data.shape[0] > max_sample_show:
        d = kde(data.T)
        prob = (1 / d) ** scatter_sample_order
        prob /= prob.sum()
        idx = np.random.choice(np.arange(data.shape[0]), size=max_sample_show, replace=False, p=prob)
        sample_show = data[idx,...]
    else:
        sample_show = data
    z = kde(sample_show.T)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))

    _default_kwargs = dict(cmap='GnBu', vmin=-0.5, vmax=1, alpha=0.8, s=10)
    _default_kwargs.update(kwargs)
    ax.scatter(
        sample_show[:, 0], sample_show[:, 1],  marker='.', c=z, 
        **_default_kwargs
    )
    return ax


def plot_contact_map(contact_rate, min_contact_rate=1e-3, log_scale=True, cmap='GnBu', bad_color='darkred', ax=None):
    """Plot contact map"""
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    contact_rate = contact_rate.copy()
    masked_contact_map = np.ma.array(contact_rate, mask=contact_rate < min_contact_rate)
    if log_scale:
        masked_contact_map = np.log(masked_contact_map, where=~np.ma.is_masked(masked_contact_map))

    cmap = plt.get_cmap(cmap)
    cmap.set_bad(bad_color, alpha=1.0)
    ax.pcolormesh(masked_contact_map, cmap=cmap)
    ax.invert_yaxis()
    ax.grid(False)
    return ax



def discretize_and_compute_jensen_shannon_dist(
    vals: np.ndarray, 
    h_ref: np.ndarray, bins: Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    min_hits: int = 1, pseudo_count: float = 1e-6
) -> Tuple[float, float]:
    """Discretize values w.r.t. ref bins and compute the Jensen-Shannon distance to reference distributions

    Only support 1D and 2D distributions

    Args:
        vals (np.ndarray): 1D or 2D value array
        h_ref (np.ndarray): 1D or 2D reference count array. 
            NOTE: for 2D array, first dim is y and second dim is x (for visualization purpose)
        bins (np.ndarray or 2-tuple of np.ndarray): bins from reference data
        compute_prec_recall (bool): if also compute the precision and recall of predicting the correct bin. 
            Positive hit is defined as >= min_hits
        pseudo_count (float): a pseudo count to add to all bins (normalized)
    
    Returns:
        JS distance and fraction of data inside the reference bins region
    """

    if len(h_ref.shape) == 1:
        assert len(vals.shape) == 1
        h_test, bins2 = np.histogram(vals, bins=bins)
        np.testing.assert_almost_equal(bins, bins2)
    elif len(h_ref.shape) == 2:
        assert len(vals.shape) == 2 and vals.shape[1] == 2
        assert isinstance(bins, tuple) and len(bins) == 2
        H_test, xedges, yedges = np.histogram2d(vals[:, 0], vals[:, 1], bins=[bins[0], bins[1]])
        H_test = H_test.transpose()
        np.testing.assert_almost_equal(bins[0], xedges)
        np.testing.assert_almost_equal(bins[1], yedges)
        # flatten: 2D -> 1D
        h_ref = h_ref.reshape(-1)
        h_test = H_test.reshape(-1)
    else:
        raise NotImplemented('histogram with d > 2 is not supported')
    
    pred = h_test >= min_hits
    gt = h_ref >= min_hits

    # per sample hit and recovery rate
    hit_rate = (h_test * gt).sum() / len(vals)
    recovery_rate = (h_ref * pred).sum() / h_ref.sum()
     
    # NOTE: this is the prec/recall interms of grids
    precision = (pred & gt).sum() / pred.sum()
    recall = (pred & gt).sum() / gt.sum()
    f1 = 2 * precision * recall / (precision + recall)

    return jensenshannon(h_test + pseudo_count, h_ref + pseudo_count), hit_rate, recovery_rate, \
        precision, recall, f1  # type:ignore


def _worker_compute_avg_js_over_dims(args):
    dist_per_1d, ref_H_1d, ref_bins_1d = args
    return discretize_and_compute_jensen_shannon_dist(dist_per_1d, ref_H_1d, ref_bins_1d)


def compute_avg_js_over_dims(dist, ref_H, ref_bins, n_proc=None):
    """Compute the average JS distance between each marginal distributions"""

    assert dist.shape[1] == ref_H.shape[0], f"dist and ref_H shape mismatch: {dist.shape} vs {ref_H.shape}"
    args_list = [(dist[:, i], ref_H[i], ref_bins[i]) for i in range(dist.shape[1])]
    result_per_dim = mp_imap(func=_worker_compute_avg_js_over_dims, iter=args_list, n_proc=n_proc, mute_tqdm=True)
    js = np.mean([res[0] for res in result_per_dim])
    hit_rate =  np.mean([res[1] for res in result_per_dim])
    return js, hit_rate


def compute_js_pwd(pdist, pair_idx, ref_dist, n_proc=None):
    """Compute JS-PwD per pairwise distance channel and then take average"""

    pdist_js, _ = filter_CA_pairwise_dist(pdist, pair_idx, excluded_neighbors=3)
    ref_H, ref_bins = ref_dist['H'], ref_dist['bins']
    return compute_avg_js_over_dims(pdist_js, ref_H, ref_bins, n_proc=n_proc)


def compute_js_tic_2d(tica_proj, ref_dist):
    """Compute JS-TIC"""
    return discretize_and_compute_jensen_shannon_dist(
        tica_proj, h_ref=ref_dist['H'], bins=(ref_dist['xedges'], ref_dist['yedges']),
    )

def compute_js_tic_1d(tica_proj, ref_dist, n_proc=None):
    """Compute JS-TIC as the average of each dim"""
    ref_H, ref_bins = ref_dist['H'], ref_dist['bins']
    return compute_avg_js_over_dims(tica_proj, ref_H, ref_bins, n_proc=1)


def compute_js_rg(rg, ref_dist, n_proc=None):
    """Compute JS-Rg per distance channel and then take average"""
    ref_H, ref_bins = ref_dist['H'], ref_dist['bins']
    return compute_avg_js_over_dims(rg, ref_H, ref_bins, n_proc=n_proc)

