"""
data_utils.py

Loading and parsing of TGA export CSVs (TA Instruments-style: Time, weight,
program/sample temperature, purge flows), per-cycle conversion / capacity
calculations, and CSV/Excel export helpers.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

M_CO2 = 44.01   # g/mol
M_CAO = 56.08   # g/mol
THEORETICAL_MASS_GAIN_FRACTION = M_CO2 / M_CAO  # ~0.7848 g CO2 / g CaO at full conversion


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _find_col(columns, must_contain, must_not_contain=()):
    for c in columns:
        cl = c.strip().lower()
        if all(tok in cl for tok in must_contain) and not any(tok in cl for tok in must_not_contain):
            return c
    return None


def load_tga_csv(file) -> pd.DataFrame:
    """Load a TGA CSV export and return a DataFrame with standardized columns:
    Time_min, Weight_mg, ProgramTemp_C, SampleTemp_C (the latter if present).

    Accepts a path, a file-like object, or bytes.
    """
    df = pd.read_csv(file)
    df.columns = [str(c).strip() for c in df.columns]
    # drop fully-empty trailing columns (TA exports often have a trailing comma)
    df = df.loc[:, ~df.columns.str.fullmatch("")]
    df = df.dropna(axis=1, how="all")

    time_col = _find_col(df.columns, ["time"])
    weight_col = _find_col(df.columns, ["weight"], must_not_contain=["baseline"])
    prog_temp_col = _find_col(df.columns, ["program", "temp"]) or _find_col(df.columns, ["temp"])
    samp_temp_col = _find_col(df.columns, ["sample", "temp"])

    if time_col is None or weight_col is None:
        raise ValueError(
            f"Could not find Time / Weight columns in file. Found columns: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "Time_min": pd.to_numeric(df[time_col], errors="coerce"),
        "Weight_mg": pd.to_numeric(df[weight_col], errors="coerce"),
    })
    if prog_temp_col is not None:
        out["ProgramTemp_C"] = pd.to_numeric(df[prog_temp_col], errors="coerce")
    if samp_temp_col is not None and samp_temp_col != prog_temp_col:
        out["SampleTemp_C"] = pd.to_numeric(df[samp_temp_col], errors="coerce")

    out = out.dropna(subset=["Time_min", "Weight_mg"]).reset_index(drop=True)
    return out


# --------------------------------------------------------------------------
# Cycle definition & metrics
# --------------------------------------------------------------------------

@dataclass
class Cycle:
    label: str
    t_start: float
    t_end: float
    source_file: str = ""
    cycle_index: int = 0


@dataclass
class CycleResult:
    label: str
    t_start: float
    t_end: float
    t_rel: np.ndarray          # time from carbonation start (min)
    X: np.ndarray               # fractional conversion (dimensionless, 0..~1)
    weight_mg: np.ndarray
    w0_mg: float
    delta_w_final_mg: float
    capacity_mmol_per_g: float  # CO2 capture capacity, mmol CO2 / g sorbent
    f_cao: float
    cycle_index: int = 0
    onset_trim_min: float = 0.0  # how much of the raw selection was trimmed as flat lead-in


def find_conversion_onset(X_raw: np.ndarray, dead_time_frac: float = 0.02) -> int:
    """Find the first index where the (provisional) conversion curve crosses
    `dead_time_frac` of that cycle's own maximum conversion - i.e. where the
    sample has actually started reacting with CO2, as opposed to the flat
    "dead time" before injection where the mass is stable. Returns 0 if
    dead_time_frac <= 0, or if the curve never clearly rises (nothing to
    trim / degenerate cycle)."""
    if dead_time_frac <= 0:
        return 0
    X_raw = np.asarray(X_raw, dtype=float)
    X_max = float(np.max(X_raw))
    if X_max <= 0:
        return 0
    threshold = dead_time_frac * X_max
    idx = np.argmax(X_raw >= threshold)
    if X_raw[idx] < threshold:
        return 0  # never crosses - leave as-is rather than trimming everything
    return int(idx)


def extract_cycle(df: pd.DataFrame, cycle: Cycle, f_cao: float = 1.0,
                   weight_units: str = "mg", dead_time_frac: float = 0.02) -> CycleResult:
    """Slice df to [t_start, t_end], compute conversion and capacity.

    `dead_time_frac` (e.g. 0.02 = 2%) ignores the pre-reaction "dead time":
    a provisional conversion curve is computed using the raw selection's
    first point as w0, the first point where that conversion crosses
    dead_time_frac * (that cycle's own max conversion) is located, everything
    before it is dropped, and t=0 / w0 are reset to that point - so even if
    the box-selection started a little early while the sample was still
    flat, the fit only ever sees the real carbonation curve. Set to 0 to
    disable and use the selection exactly as drawn."""
    seg = df[(df["Time_min"] >= cycle.t_start) & (df["Time_min"] <= cycle.t_end)].copy()
    seg = seg.sort_values("Time_min").reset_index(drop=True)
    if len(seg) < 2:
        raise ValueError(f"Cycle '{cycle.label}': fewer than 2 points in selected range.")

    unit_factor = 1.0 if weight_units == "mg" else 1000.0  # µg -> mg
    w = seg["Weight_mg"].values / unit_factor
    t_rel = (seg["Time_min"].values - seg["Time_min"].values[0])

    onset_trim_min = 0.0
    if dead_time_frac > 0:
        w0_raw = float(w[0])
        denom_raw = w0_raw * max(f_cao, 1e-9) * THEORETICAL_MASS_GAIN_FRACTION
        X_raw = (w - w0_raw) / denom_raw if denom_raw > 0 else np.zeros_like(w)
        onset_idx = find_conversion_onset(X_raw, dead_time_frac)
        if onset_idx > 0:
            onset_trim_min = float(t_rel[onset_idx])
            w = w[onset_idx:]
            t_rel = t_rel[onset_idx:] - t_rel[onset_idx]

    w0 = float(w[0])
    delta_w = w - w0
    delta_w_final = float(delta_w[-1])

    # headline metric: mmol CO2 captured per gram of sorbent (direct mass-gain basis)
    capacity_mmol_per_g = (delta_w_final / M_CO2) * 1000.0 / w0 if w0 != 0 else np.nan

    # fractional conversion for kinetic fitting, normalized to the theoretical
    # maximum mass gain for the (user-specified) active CaO fraction
    denom = w0 * max(f_cao, 1e-9) * THEORETICAL_MASS_GAIN_FRACTION
    X = delta_w / denom if denom > 0 else np.zeros_like(delta_w)
    X = np.clip(X, 0.0, None)

    return CycleResult(
        label=cycle.label, t_start=cycle.t_start, t_end=cycle.t_end,
        t_rel=t_rel, X=X, weight_mg=w, w0_mg=w0,
        delta_w_final_mg=delta_w_final, capacity_mmol_per_g=capacity_mmol_per_g,
        f_cao=f_cao, cycle_index=cycle.cycle_index, onset_trim_min=onset_trim_min,
    )


# --------------------------------------------------------------------------
# Convenience: auto-detect candidate carbonation segments from a temperature
# plateau (optional helper; manual drag-selection on the chart remains the
# primary, always-available workflow).
# --------------------------------------------------------------------------

def detect_plateau_segments(df: pd.DataFrame, target_temp: float, tolerance: float = 5.0,
                             min_duration_min: float = 2.0) -> list[tuple[float, float]]:
    """Return a list of (t_start, t_end) for contiguous stretches where
    ProgramTemp_C stays within `tolerance` of `target_temp`, dropping any
    stretch shorter than `min_duration_min`."""
    if "ProgramTemp_C" not in df.columns:
        raise ValueError("This file has no ProgramTemp_C column; use manual selection instead.")

    t = df["Time_min"].values
    temp = df["ProgramTemp_C"].values
    mask = np.abs(temp - target_temp) <= tolerance
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []

    segments = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i != prev + 1:
            segments.append((start, prev))
            start = i
        prev = i
    segments.append((start, prev))

    out = []
    for s, e in segments:
        if t[e] - t[s] >= min_duration_min:
            out.append((float(t[s]), float(t[e])))
    return out


# --------------------------------------------------------------------------
# Export helpers
# --------------------------------------------------------------------------

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dfs_to_excel_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            safe_name = str(name)[:31]
            df.to_excel(writer, sheet_name=safe_name, index=False)
    return buf.getvalue()
