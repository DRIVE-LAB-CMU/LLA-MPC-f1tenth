"""
find_top_models.py
==================
Finds the most frequently selected model indices in a LLA log and reports
the corresponding parameter sets in a form ready to paste into the
`general_models` dict of the visualizer.
"""

import os
import numpy as np
from collections import Counter


# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------
NPZ_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nomf4.npz")
TOP_N      = 5            # how many top models to report
USE_MEDIAN = False        # True = use median params, False = use mean

# Must match the visualizer's DEFAULT_LOG_ORDER (order params are stored in npz)
# LOG_ORDER  = ['Bf', 'Br', 'Cf', 'Cr', 'Df', 'Dr', 'Cro', 'Cd', 'Ce', 'Cm']
LOG_ORDER  = ['Cf', 'Cr', 'muf','mur','Cro']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_npz(path):
    data = np.load(path, allow_pickle=True)

    model_index = np.asarray(data["model_index"])

    # params: (T, P) where P = number of Pacejka params
    params_raw = data["params"]
    if isinstance(params_raw, np.ndarray) and params_raw.ndim == 2:
        # Determine orientation: rows = timesteps or rows = params?
        if params_raw.shape[0] == len(model_index):
            params = params_raw                   # (T, P)
        else:
            params = params_raw.T                 # (P, T) -> (T, P)
    else:
        # Ragged or object array — stack into (T, P)
        params = np.stack([np.asarray(p, dtype=float) for p in params_raw])

    return model_index, params.astype(float)


def find_top_models(model_index, params, log_order, top_n):
    """
    Returns a list of dicts (sorted by selection count, descending):
        {
          'model_idx':  int,
          'count':      int,
          'fraction':   float,
          'mean':       {param_name: float},
          'std':        {param_name: float},
          'median':     {param_name: float},
        }
    """
    counts = Counter(model_index.tolist())
    total  = len(model_index)

    results = []
    for idx, count in counts.most_common(top_n):
        mask   = model_index == idx
        subset = params[mask]                     # (count, P)

        n_params = min(subset.shape[1], len(log_order))
        mean   = subset[:, :n_params].mean(axis=0)
        std    = subset[:, :n_params].std(axis=0)
        median = np.median(subset[:, :n_params], axis=0)

        results.append({
            "model_idx": int(idx),
            "count":     count,
            "fraction":  count / total,
            "mean":      dict(zip(log_order[:n_params], mean)),
            "std":       dict(zip(log_order[:n_params], std)),
            "median":    dict(zip(log_order[:n_params], median)),
        })

    return results


def print_table(results, log_order):
    col_w  = 10
    header = f"{'Rank':<5} {'Idx':>5} {'Count':>7} {'Frac%':>7}  " + \
             "  ".join(f"{k:>{col_w}}" for k in log_order)
    print(header)
    print("-" * len(header))

    for rank, r in enumerate(results, 1):
        vals = "  ".join(f"{r['mean'].get(k, 0.0):>{col_w}.4f}" for k in log_order)
        print(f"{rank:<5} {r['model_idx']:>5} {r['count']:>7} {r['fraction']*100:>6.1f}%  {vals}")


def print_general_models_block(results, log_order, use_median=False):
    """Print a general_models dict ready to paste into the visualizer."""
    key = "median" if use_median else "mean"
    print(f"\n# --- general_models (top-{len(results)} by selection count,"
          f" params = {key} over selected timesteps) ---")
    print("general_models = {")
    for rank, r in enumerate(results, 1):
        name = f"model_{r['model_idx']}"
        frac = r['fraction'] * 100
        print(f"    # rank {rank}  |  model index {r['model_idx']}"
              f"  |  selected {r['count']}x ({frac:.1f}%)")
        print(f"    \"{name}\": {{")
        for k in log_order:
            v = r[key].get(k, 0.0)
            s = r["std"].get(k, 0.0)
            print(f"        '{k}': {v:.6f},  # std={s:.4f}")
        print("    },")
    print("}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\nLoading: {NPZ_PATH}")
    model_index, params = load_npz(NPZ_PATH)
    print(f"  Timesteps     : {len(model_index)}")
    print(f"  Param dim     : {params.shape[1]}")
    print(f"  Unique models : {len(set(model_index.tolist()))}")

    results = find_top_models(model_index, params, LOG_ORDER, TOP_N)

    agg = "median" if USE_MEDIAN else "mean"
    print(f"\n=== Top {len(results)} most-selected models ({agg} params) ===\n")
    print_table(results, LOG_ORDER)
    print_general_models_block(results, LOG_ORDER, use_median=USE_MEDIAN)