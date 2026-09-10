"""
kinetic_models.py

Kinetic models for CaO / calcium-looping carbonation curves, plus a generic
two-stage (fast reaction-controlled + slow diffusion-controlled) fitting
engine with per-model optimized transition point, and AIC/BIC/R2 scoring.

All models are expressed as explicit fractional-conversion functions
    X = f(t; params)
with X(0) = 0, so that the *same* functional form can be re-used, re-fit
with its own rate constant, on the "remaining" conversion after a
transition point (see fit_two_stage).

Model list (standard models used in the calcium-looping / gas-solid
carbonation literature):
    zero_order            - reaction control, linear (planar) geometry
    first_order            - apparent first-order (most widely used fast-stage model)
    second_order           - apparent second-order
    nth_order               - apparent n-th order (n fitted)
    avrami_erofeev          - Avrami-Erofeev / JMAK nucleation-growth model
    scm_sphere               - Shrinking Core Model, reaction control, sphere
    scm_cylinder             - Shrinking Core Model, reaction control, cylinder
    jander                     - Jander diffusion model (product-layer diffusion, sphere)
    ginstling_brounshtein     - Ginstling-Brounshtein diffusion model (sphere)
    parabolic                  - Parabolic diffusion law (planar)
    random_pore_model (RPM)   - Bhatia-Perlmutter random pore model
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import curve_fit


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _safe_clip(X):
    return np.clip(X, 0.0, 1.0)


def _gb_inverse(kt, lo=0.0, hi=1.0 - 1e-9, n_iter=60):
    """Vectorized bisection inversion of the Ginstling-Brounshtein equation
    g(X) = 1 - (2/3)X - (1-X)^(2/3) = kt  for X, given kt (scalar or array).
    g is monotonically increasing on [0, hi]; g(0)=0, g(hi) = g_hi."""
    kt = np.atleast_1d(np.asarray(kt, dtype=float))
    g_hi = 1.0 - (2.0 / 3.0) * hi - (1.0 - hi) ** (2.0 / 3.0)
    kt_c = np.clip(kt, 0.0, g_hi)

    lo_arr = np.zeros_like(kt_c)
    hi_arr = np.full_like(kt_c, hi)
    for _ in range(n_iter):
        mid = 0.5 * (lo_arr + hi_arr)
        g_mid = 1.0 - (2.0 / 3.0) * mid - (1.0 - mid) ** (2.0 / 3.0)
        too_low = g_mid < kt_c
        lo_arr = np.where(too_low, mid, lo_arr)
        hi_arr = np.where(too_low, hi_arr, mid)
    out = 0.5 * (lo_arr + hi_arr)
    out = np.where(kt <= 0, 0.0, out)
    out = np.where(kt >= g_hi, hi, out)
    return out


# --------------------------------------------------------------------------
# Model forward functions:  X = f(t, *params)
# --------------------------------------------------------------------------

def f_zero_order(t, k):
    return _safe_clip(k * t)


def f_first_order(t, k):
    k = max(k, 1e-12)
    return _safe_clip(1.0 - np.exp(-k * t))


def f_second_order(t, k):
    k = max(k, 1e-12)
    kt = k * t
    return _safe_clip(kt / (1.0 + kt))


def f_nth_order(t, k, n):
    k = max(k, 1e-12)
    if abs(n - 1.0) < 1e-6:
        return f_first_order(t, k)
    base = 1.0 - (1.0 - n) * k * t
    base = np.where(base > 0, base, 0.0)
    X = 1.0 - base ** (1.0 / (1.0 - n))
    return _safe_clip(np.nan_to_num(X, nan=1.0, posinf=1.0, neginf=0.0))


def f_avrami_erofeev(t, k, n):
    k = max(k, 1e-12)
    n = max(n, 1e-3)
    kt = np.clip(k * t, 0, None)
    return _safe_clip(1.0 - np.exp(-(kt ** n)))


def f_scm_sphere(t, k):
    kt = np.clip(k * t, 0.0, 1.0)
    return _safe_clip(1.0 - (1.0 - kt) ** 3)


def f_scm_cylinder(t, k):
    kt = np.clip(k * t, 0.0, 1.0)
    return _safe_clip(1.0 - (1.0 - kt) ** 2)


def f_jander(t, k):
    kt = np.clip(k * t, 0.0, 1.0)
    return _safe_clip(1.0 - (1.0 - np.sqrt(kt)) ** 3)


def f_ginstling_brounshtein(t, k):
    k = max(k, 1e-12)
    return _safe_clip(_gb_inverse(k * np.asarray(t, dtype=float)))


def f_parabolic(t, k):
    kt = np.clip(k * np.asarray(t, dtype=float), 0.0, None)
    return _safe_clip(np.sqrt(kt))


def f_random_pore_model(t, k, psi):
    k = max(k, 1e-12)
    psi = max(psi, 1e-6)
    kt = np.clip(k * np.asarray(t, dtype=float), 0.0, None)
    inner = np.clip(1.0 + psi * kt, 0.0, None)
    return _safe_clip(1.0 - np.exp((2.0 / psi) * (1.0 - np.sqrt(inner))))


# --------------------------------------------------------------------------
# Model registry
# --------------------------------------------------------------------------

@dataclass
class KineticModel:
    key: str
    label: str
    stage_type: str  # "reaction" or "diffusion" or "structural"
    func: Callable
    param_names: Sequence[str]
    p0: Callable[[np.ndarray, np.ndarray], list]
    bounds: tuple

    def n_params(self):
        return len(self.param_names)


def _rep_point(t, X, x_cap=0.95, frac=0.5):
    """Pick a representative interior data point (away from t=0 and away
    from saturation near X=1) to compute a quick closed-form estimate of a
    model's rate constant, so curve_fit starts close to the solution and
    converges in far fewer iterations."""
    n = len(t)
    if n == 0:
        return 1.0, 0.5
    start = int(np.clip(frac * n, 1, max(n - 1, 1)))
    order = list(range(start, n)) + list(range(start - 1, -1, -1))
    for i in order:
        if t[i] > 1e-9 and 0.0 < X[i] < x_cap:
            return float(t[i]), float(X[i])
    tmax = max(float(np.max(t)), 1e-6)
    return tmax, float(min(max(np.max(X), 1e-3), x_cap))


def _p0_k_only(t, X):
    tmax = max(float(np.max(t)), 1e-6)
    return [1.0 / tmax]


def _p0_zero_order(t, X):
    tm, Xm = _rep_point(t, X, x_cap=0.999)
    return [max(Xm / tm, 1e-6)]


def _p0_first_order(t, X):
    tm, Xm = _rep_point(t, X, x_cap=0.95)
    return [max(-np.log(1.0 - Xm) / tm, 1e-6)]


def _p0_second_order(t, X):
    tm, Xm = _rep_point(t, X, x_cap=0.95)
    return [max((Xm / (1.0 - Xm)) / tm, 1e-6)]


def _p0_scm_sphere(t, X):
    tm, Xm = _rep_point(t, X, x_cap=0.999)
    return [max((1.0 - (1.0 - Xm) ** (1.0 / 3.0)) / tm, 1e-6)]


def _p0_scm_cylinder(t, X):
    tm, Xm = _rep_point(t, X, x_cap=0.999)
    return [max((1.0 - (1.0 - Xm) ** 0.5) / tm, 1e-6)]


def _p0_jander(t, X):
    tm, Xm = _rep_point(t, X, x_cap=0.999)
    return [max((1.0 - (1.0 - Xm) ** (1.0 / 3.0)) ** 2 / tm, 1e-6)]


def _p0_gb(t, X):
    tm, Xm = _rep_point(t, X, x_cap=0.999)
    g = 1.0 - (2.0 / 3.0) * Xm - (1.0 - Xm) ** (2.0 / 3.0)
    return [max(g / tm, 1e-6)]


def _p0_parabolic(t, X):
    tm, Xm = _rep_point(t, X, x_cap=0.999)
    return [max((Xm ** 2) / tm, 1e-6)]


def _p0_k_n(t, X):
    tm, Xm = _rep_point(t, X, x_cap=0.95)
    return [max(-np.log(1.0 - Xm) / tm, 1e-6), 1.0]


def _p0_k_psi(t, X):
    tm, Xm = _rep_point(t, X, x_cap=0.95)
    return [max(-np.log(1.0 - Xm) / tm, 1e-6), 1.0]


MODELS: dict[str, KineticModel] = {
    "zero_order": KineticModel(
        "zero_order", "Zero-order (reaction control, planar)", "reaction",
        f_zero_order, ["k"], _p0_zero_order, ([1e-8], [1e3]),
    ),
    "first_order": KineticModel(
        "first_order", "Apparent first-order", "reaction",
        f_first_order, ["k"], _p0_first_order, ([1e-8], [1e3]),
    ),
    "second_order": KineticModel(
        "second_order", "Apparent second-order", "reaction",
        f_second_order, ["k"], _p0_second_order, ([1e-8], [1e3]),
    ),
    "nth_order": KineticModel(
        "nth_order", "Apparent n-th order", "reaction",
        f_nth_order, ["k", "n"], _p0_k_n, ([1e-8, 0.1], [1e3, 5.0]),
    ),
    "avrami_erofeev": KineticModel(
        "avrami_erofeev", "Avrami-Erofeev (JMAK, nucleation-growth)", "reaction",
        f_avrami_erofeev, ["k", "n"], _p0_k_n, ([1e-8, 0.1], [1e3, 5.0]),
    ),
    "scm_sphere": KineticModel(
        "scm_sphere", "Shrinking Core Model - reaction control (sphere)", "reaction",
        f_scm_sphere, ["k"], _p0_scm_sphere, ([1e-8], [1e3]),
    ),
    "scm_cylinder": KineticModel(
        "scm_cylinder", "Shrinking Core Model - reaction control (cylinder)", "reaction",
        f_scm_cylinder, ["k"], _p0_scm_cylinder, ([1e-8], [1e3]),
    ),
    "jander": KineticModel(
        "jander", "Jander diffusion model (sphere)", "diffusion",
        f_jander, ["k"], _p0_jander, ([1e-8], [1e3]),
    ),
    "ginstling_brounshtein": KineticModel(
        "ginstling_brounshtein", "Ginstling-Brounshtein diffusion model (sphere)", "diffusion",
        f_ginstling_brounshtein, ["k"], _p0_gb, ([1e-8], [1e3]),
    ),
    "parabolic": KineticModel(
        "parabolic", "Parabolic diffusion law (planar)", "diffusion",
        f_parabolic, ["k"], _p0_parabolic, ([1e-8], [1e3]),
    ),
    "random_pore_model": KineticModel(
        "random_pore_model", "Random Pore Model (Bhatia-Perlmutter)", "structural",
        f_random_pore_model, ["k", "psi"], _p0_k_psi, ([1e-8, 1e-4], [1e3, 200.0]),
    ),
}

MODEL_ORDER = list(MODELS.keys())


# --------------------------------------------------------------------------
# Fit statistics
# --------------------------------------------------------------------------

def aic_bic(n: int, k: int, rss: float):
    """n = number of data points, k = number of free parameters, rss = residual sum of squares."""
    rss = max(rss, 1e-15)
    aic = n * np.log(rss / n) + 2 * k
    bic = n * np.log(rss / n) + k * np.log(n)
    return float(aic), float(bic)


def r_squared(X_obs, X_pred):
    X_obs = np.asarray(X_obs, dtype=float)
    X_pred = np.asarray(X_pred, dtype=float)
    ss_res = np.sum((X_obs - X_pred) ** 2)
    ss_tot = np.sum((X_obs - np.mean(X_obs)) ** 2)
    if ss_tot <= 1e-15:
        return 1.0 if ss_res <= 1e-15 else 0.0
    return float(1.0 - ss_res / ss_tot)


def downsample_series(t, X, max_points: int = 250):
    """Evenly subsample (t, X) down to at most max_points actual data points
    (always keeping the first and last point). TGA curves are smooth, so
    fitting to an even subsample gives essentially the same parameters/R2 as
    fitting to every point, at a fraction of the compute cost - this matters
    because the two-stage fit runs a breakpoint grid search."""
    t = np.asarray(t, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(t)
    if n <= max_points:
        return t, X
    idx = np.unique(np.linspace(0, n - 1, max_points).astype(int))
    return t[idx], X[idx]


def evaluate_two_stage_curve(t_query, breakpoint_t: float | None,
                              model_key1: str, stage1_params: dict,
                              model_key2: str | None = None, stage2_params: dict | None = None):
    """Reconstruct the fitted (piecewise) conversion curve at arbitrary query
    times, given the fitted stage parameters - used to plot a smooth curve
    against the full-resolution data even when fitting was done on a
    downsampled subset."""
    t_query = np.asarray(t_query, dtype=float)
    model1 = MODELS[model_key1]

    if breakpoint_t is None or stage2_params is None:
        return model1.func(t_query, *[stage1_params[p] for p in model1.param_names])

    model2 = MODELS[model_key2 or model_key1]
    tb = breakpoint_t
    Xb = float(model1.func(np.array([tb]), *[stage1_params[p] for p in model1.param_names])[0])

    X_out = np.empty_like(t_query)
    mask1 = t_query <= tb
    mask2 = ~mask1
    if np.any(mask1):
        X_out[mask1] = model1.func(t_query[mask1], *[stage1_params[p] for p in model1.param_names])
    if np.any(mask2):
        t2_local = t_query[mask2] - tb
        X2_local = model2.func(t2_local, *[stage2_params[p] for p in model2.param_names])
        X_out[mask2] = Xb + (1.0 - Xb) * X2_local
    return X_out


def evaluate_fit_result_curve(t_query, fit_result: "FitResult", model_key1: str,
                               model_key2: str | None = None):
    """Convenience wrapper around evaluate_two_stage_curve that pulls the
    right parameter dict out of a FitResult (handles both the normal
    two-stage case and the too-few-points single-stage fallback)."""
    if not fit_result.success:
        return np.full(len(np.asarray(t_query)), np.nan)
    if fit_result.breakpoint_t is None:
        return evaluate_two_stage_curve(t_query, None, model_key1, fit_result.params)
    return evaluate_two_stage_curve(
        t_query, fit_result.breakpoint_t, model_key1, fit_result.stage1_params,
        model_key2, fit_result.stage2_params,
    )


# --------------------------------------------------------------------------
# Single-stage fit
# --------------------------------------------------------------------------

@dataclass
class FitResult:
    model_key: str
    params: dict
    X_pred: np.ndarray
    rss: float
    r2: float
    aic: float
    bic: float
    n_points: int
    n_params: int
    success: bool
    breakpoint_t: float | None = None
    breakpoint_idx: int | None = None
    stage1_params: dict | None = None
    stage2_params: dict | None = None
    r2_stage1: float | None = None
    r2_stage2: float | None = None


def fit_single_stage(t, X, model_key: str) -> FitResult:
    model = MODELS[model_key]
    t = np.asarray(t, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(t)
    k = model.n_params()

    if n <= k:
        return FitResult(model_key, {}, np.full(n, np.nan), np.inf, -np.inf,
                          np.inf, np.inf, n, k, False)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                model.func, t, X, p0=model.p0(t, X),
                bounds=model.bounds, maxfev=3000,
            )
        X_pred = model.func(t, *popt)
        rss = float(np.sum((X - X_pred) ** 2))
        r2 = r_squared(X, X_pred)
        aic, bic = aic_bic(n, k, rss)
        params = dict(zip(model.param_names, popt))
        return FitResult(model_key, params, X_pred, rss, r2, aic, bic, n, k, True)
    except Exception:
        return FitResult(model_key, {}, np.full(n, np.nan), np.inf, -np.inf,
                          np.inf, np.inf, n, k, False)


# --------------------------------------------------------------------------
# Two-stage fit (fast reaction stage + slow diffusion stage), with an
# optimized transition point specific to each model.
# --------------------------------------------------------------------------

def _fit_stage(t_local, X_local, model: KineticModel):
    """Fit a model to a single segment where t_local starts at 0 and X_local
    starts at 0 (already renormalized for stage 2 if needed)."""
    n = len(t_local)
    k = model.n_params()
    if n <= k:
        return None, np.inf, np.full(n, np.nan)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                model.func, t_local, X_local, p0=model.p0(t_local, X_local),
                bounds=model.bounds, maxfev=3000,
            )
        X_pred = model.func(t_local, *popt)
        rss = float(np.sum((X_local - X_pred) ** 2))
        return popt, rss, X_pred
    except Exception:
        return None, np.inf, np.full(n, np.nan)


def fit_two_stage(t, X, model_key: str, min_frac: float = 0.08,
                   min_pts: int = 6, coarse_candidates: int = 120) -> FitResult:
    """Fit the same kinetic model separately to a fast stage [0, tb] and a
    slow stage (tb, tmax], choosing the transition point tb (per this model)
    that minimizes the combined residual sum of squares.

    Stage 2 is fit on the *remaining* conversion, renormalized to [0, ~1]:
        X2_local(t2) = (X(t) - X(tb)) / (1 - X(tb)),   t2 = t - tb
    so the same functional forms (all of which satisfy f(0)=0) can be
    reused with their own, independently fitted rate constant.
    """
    model = MODELS[model_key]
    t = np.asarray(t, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(t)
    k_stage = model.n_params()
    n_params_total = 2 * k_stage + 1  # +1 for the fitted breakpoint itself

    min_side = max(min_pts, k_stage + 2)
    lo = min_side
    hi = n - min_side
    # also respect a minimum fraction of the curve on each side
    lo = max(lo, int(min_frac * n))
    hi = min(hi, n - int(min_frac * n))

    if hi <= lo or n < 2 * min_side + 1:
        # not enough points to split; fall back to single-stage fit on whole curve
        single = fit_single_stage(t, X, model_key)
        single.breakpoint_t = None
        return single

    # coarse grid over candidate breakpoint indices
    step = max(1, (hi - lo) // coarse_candidates)
    candidates = list(range(lo, hi + 1, step))
    if candidates[-1] != hi:
        candidates.append(hi)

    def eval_breakpoint(j):
        t1, X1 = t[: j + 1], X[: j + 1]
        tb = t[j]
        Xb = X[j]
        t2_raw, X2_raw = t[j:], X[j:]
        t2 = t2_raw - tb
        denom = max(1.0 - Xb, 1e-9)
        X2 = np.clip((X2_raw - Xb) / denom, 0.0, None)

        p1, rss1, Xp1 = _fit_stage(t1, X1, model)
        p2, rss2, Xp2_local = _fit_stage(t2, X2, model)
        if p1 is None or p2 is None:
            return np.inf, None
        Xp2 = Xb + denom * Xp2_local
        rss_total = float(np.sum((X1 - Xp1) ** 2)) + float(np.sum((X2_raw - Xp2) ** 2))
        info = dict(j=j, tb=tb, Xb=Xb, p1=p1, p2=p2, Xp1=Xp1, Xp2=Xp2,
                    rss1=float(np.sum((X1 - Xp1) ** 2)),
                    rss2=float(np.sum((X2_raw - Xp2) ** 2)))
        return rss_total, info

    best_rss, best_info = np.inf, None
    for j in candidates:
        rss_total, info = eval_breakpoint(j)
        if rss_total < best_rss:
            best_rss, best_info = rss_total, info

    if best_info is None:
        single = fit_single_stage(t, X, model_key)
        single.breakpoint_t = None
        return single

    # refine locally around the best coarse candidate
    j_center = best_info["j"]
    refine_lo = max(lo, j_center - step)
    refine_hi = min(hi, j_center + step)
    for j in range(refine_lo, refine_hi + 1):
        rss_total, info = eval_breakpoint(j)
        if rss_total < best_rss:
            best_rss, best_info = rss_total, info

    j = best_info["j"]
    X_pred = np.concatenate([best_info["Xp1"][:-1], best_info["Xp2"]])
    # Xp1 and Xp2 both include the breakpoint sample once each; drop the
    # duplicate from stage 1 so X_pred has length n and matches t/X exactly.
    rss_total = best_rss
    r2 = r_squared(X, X_pred)
    aic, bic = aic_bic(n, n_params_total, rss_total)

    r2_1 = r_squared(t[: j + 1] * 0 + X[: j + 1], best_info["Xp1"])  # placeholder recompute below
    r2_1 = r_squared(X[: j + 1], best_info["Xp1"])
    r2_2 = r_squared(X[j:], best_info["Xp2"])

    return FitResult(
        model_key=model_key,
        params={},
        X_pred=X_pred,
        rss=rss_total,
        r2=r2,
        aic=aic,
        bic=bic,
        n_points=n,
        n_params=n_params_total,
        success=True,
        breakpoint_t=float(best_info["tb"]),
        breakpoint_idx=int(j),
        stage1_params=dict(zip(model.param_names, best_info["p1"])),
        stage2_params=dict(zip(model.param_names, best_info["p2"])),
        r2_stage1=r2_1,
        r2_stage2=r2_2,
    )


def fit_two_stage_mixed(t, X, model_key_stage1: str, model_key_stage2: str,
                         min_frac: float = 0.08, min_pts: int = 6,
                         coarse_candidates: int = 120) -> FitResult:
    """Like fit_two_stage, but stage 1 and stage 2 may use *different*
    kinetic models (e.g. an apparent first-order model for the fast stage
    and a Ginstling-Brounshtein diffusion model for the slow stage)."""
    model1 = MODELS[model_key_stage1]
    model2 = MODELS[model_key_stage2]
    t = np.asarray(t, dtype=float)
    X = np.asarray(X, dtype=float)
    n = len(t)
    k1, k2 = model1.n_params(), model2.n_params()
    n_params_total = k1 + k2 + 1

    min_side = max(min_pts, max(k1, k2) + 2)
    lo = max(min_side, int(min_frac * n))
    hi = min(n - min_side, n - int(min_frac * n))
    if hi <= lo or n < 2 * min_side + 1:
        return FitResult(f"{model_key_stage1}+{model_key_stage2}", {}, np.full(n, np.nan),
                         np.inf, -np.inf, np.inf, np.inf, n, n_params_total, False)

    step = max(1, (hi - lo) // coarse_candidates)
    candidates = list(range(lo, hi + 1, step))
    if candidates[-1] != hi:
        candidates.append(hi)

    def eval_breakpoint(j):
        t1, X1 = t[: j + 1], X[: j + 1]
        tb, Xb = t[j], X[j]
        t2_raw, X2_raw = t[j:], X[j:]
        t2 = t2_raw - tb
        denom = max(1.0 - Xb, 1e-9)
        X2 = np.clip((X2_raw - Xb) / denom, 0.0, None)

        p1, rss1, Xp1 = _fit_stage(t1, X1, model1)
        p2, rss2, Xp2_local = _fit_stage(t2, X2, model2)
        if p1 is None or p2 is None:
            return np.inf, None
        Xp2 = Xb + denom * Xp2_local
        rss_total = float(np.sum((X1 - Xp1) ** 2)) + float(np.sum((X2_raw - Xp2) ** 2))
        return rss_total, dict(j=j, tb=tb, Xb=Xb, p1=p1, p2=p2, Xp1=Xp1, Xp2=Xp2)

    best_rss, best_info = np.inf, None
    for j in candidates:
        rss_total, info = eval_breakpoint(j)
        if rss_total < best_rss:
            best_rss, best_info = rss_total, info

    if best_info is None:
        return FitResult(f"{model_key_stage1}+{model_key_stage2}", {}, np.full(n, np.nan),
                         np.inf, -np.inf, np.inf, np.inf, n, n_params_total, False)

    j_center = best_info["j"]
    refine_lo = max(lo, j_center - step)
    refine_hi = min(hi, j_center + step)
    for j in range(refine_lo, refine_hi + 1):
        rss_total, info = eval_breakpoint(j)
        if rss_total < best_rss:
            best_rss, best_info = rss_total, info

    j = best_info["j"]
    X_pred = np.concatenate([best_info["Xp1"][:-1], best_info["Xp2"]])
    r2 = r_squared(X, X_pred)
    aic, bic = aic_bic(n, n_params_total, best_rss)
    r2_1 = r_squared(X[: j + 1], best_info["Xp1"])
    r2_2 = r_squared(X[j:], best_info["Xp2"])

    return FitResult(
        model_key=f"{model_key_stage1}+{model_key_stage2}",
        params={},
        X_pred=X_pred,
        rss=best_rss,
        r2=r2,
        aic=aic,
        bic=bic,
        n_points=n,
        n_params=n_params_total,
        success=True,
        breakpoint_t=float(best_info["tb"]),
        breakpoint_idx=int(j),
        stage1_params=dict(zip(model1.param_names, best_info["p1"])),
        stage2_params=dict(zip(model2.param_names, best_info["p2"])),
        r2_stage1=r2_1,
        r2_stage2=r2_2,
    )


def fit_all_models_two_stage(t, X, model_keys: Sequence[str] | None = None,
                              **kwargs) -> dict[str, FitResult]:
    keys = list(model_keys) if model_keys else MODEL_ORDER
    return {key: fit_two_stage(t, X, key, **kwargs) for key in keys}


# --------------------------------------------------------------------------
# Cycle-to-cycle capacity decay: Grasa & Abanades (2006) deactivation model.
# Not a per-cycle kinetic model, but the standard way multi-cycle calcium
# looping capacity decay is fit/compared in the literature.
#   X_N = X_r + (X_0 - X_r) * F1 * (1 - F1) / (1 - F1 + N * F1 * (1 - F1)) -- (general form)
# The commonly used simplified two-parameter form is:
#   X_N = 1 / (1/(1 - X_r) + k * N) + X_r
# with X_r = residual conversion (N -> infinity), k = deactivation constant.
# --------------------------------------------------------------------------

def f_grasa_abanades(N, Xr, k):
    Xr = np.clip(Xr, 0.0, 1.0)
    k = max(k, 1e-9)
    return Xr + 1.0 / (1.0 / max(1.0 - Xr, 1e-9) + k * np.asarray(N, dtype=float))


def fit_grasa_abanades(cycle_numbers, capacities):
    """Fit the Grasa-Abanades deactivation model to capacity (or conversion)
    vs cycle number. Returns (Xr, k, r2, predict_fn) or None if the fit fails
    or there are fewer than 3 cycles."""
    N = np.asarray(cycle_numbers, dtype=float)
    Y = np.asarray(capacities, dtype=float)
    if len(N) < 3:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p0 = [max(Y.min(), 1e-6), 0.1]
            popt, _ = curve_fit(f_grasa_abanades, N, Y, p0=p0,
                                bounds=([0, 1e-6], [max(Y.max(), 1.0), 10.0]),
                                maxfev=20000)
        Y_pred = f_grasa_abanades(N, *popt)
        r2 = r_squared(Y, Y_pred)
        return {"Xr": float(popt[0]), "k": float(popt[1]), "r2": r2}
    except Exception:
        return None
