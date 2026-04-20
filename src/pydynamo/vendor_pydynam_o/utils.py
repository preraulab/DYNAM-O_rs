"""Minimal utils subset needed by the vendored TFpeaks.py.

Only `pow2db` and `min_prominence` are required for detect_tfpeaks. The
other helpers in pyDYNAM-O/dynam_o/utils.py (detect_artifacts,
summary_plot, etc.) are not used in the vendored detection path.
"""

import numpy as np
from scipy.stats import chi2


def pow2db(y):
    """Converts power to dB, NaN for non-positive values.

    Matches pyDYNAM-O's implementation including the +300/-300 rounding
    trick which guarantees integer results for exact negative powers of 10.
    """
    if isinstance(y, (int, float)):
        if y == 0:
            return np.nan
        return (10 * np.log10(y) + 300) - 300
    if isinstance(y, list):
        y = np.asarray(y)
    y = np.asarray(y, dtype=float).copy()
    y[y == 0] = np.nan
    return (10 * np.log10(y) + 300) - 300


def min_prominence(num_tapers: int, alpha: float = 0.95) -> float:
    """Chi-squared CI lower-bound for minimum peak prominence (dB)."""
    chi2_df = 2 * num_tapers
    return -pow2db(chi2_df / chi2.ppf(alpha / 2 + 0.5, chi2_df)) * 2
