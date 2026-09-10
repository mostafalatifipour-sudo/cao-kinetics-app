"""
CaO Calcium-Looping TGA Kinetics Analyzer
==========================================

A Streamlit app for analyzing thermogravimetric (TGA) data from CaO
calcium-looping cycles:

  * Drag-select each cycle's carbonation start/end directly on a chart
  * CO2 capture capacity (mmol CO2 / g sorbent) per cycle
  * Two-stage kinetic fitting (fast reaction stage + slow diffusion stage)
    with a transition point independently optimized for each of 11
    standard calcium-looping kinetic models
  * R2 / AIC / BIC comparison table across models
  * Cycle-to-cycle comparison, including a Grasa-Abanades capacity-decay fit
  * CSV / Excel export of all tables, PNG / PDF / SVG export of figures

Run locally with:   streamlit run app.py
"""

from __future__ import annotations

import io
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data_utils import (
    Cycle, load_tga_csv, extract_cycle, detect_plateau_segments,
    df_to_csv_bytes, dfs_to_excel_bytes,
)
from kinetic_models import (
    MODELS, MODEL_ORDER, fit_two_stage, fit_two_stage_mixed,
    fit_grasa_abanades, f_grasa_abanades, downsample_series,
)
from figures import (
    plot_cycle_fit, plot_cycle_comparison_curves, plot_capacity_bar,
    plot_deactivation_fit, fig_to_bytes,
)

st.set_page_config(page_title="CaO Calcium-Looping Kinetics", layout="wide")


# --------------------------------------------------------------------------
# Session state initialization
# --------------------------------------------------------------------------

def init_state():
    st.session_state.setdefault("dataframes", {})       # filename -> DataFrame
    st.session_state.setdefault("cycles", [])            # list[Cycle]
    st.session_state.setdefault("results_cache", {})      # cycle label -> {model_key: FitResult}
    st.session_state.setdefault("mixed_results_cache", {})
    st.session_state.setdefault("next_cycle_index", 1)


init_state()


def get_active_df() -> tuple[str, pd.DataFrame] | tuple[None, None]:
    files = st.session_state["dataframes"]
    if not files:
        return None, None
    fname = st.session_state.get("active_file")
    if fname not in files:
        fname = list(files.keys())[0]
    return fname, files[fname]


# --------------------------------------------------------------------------
# Sidebar: global settings
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("Settings")
    weight_units = st.radio("Raw weight column units", ["mg", "µg"], index=0,
                            help="Units of the Weight column in your TGA export. "
                                 "TA Instruments exports are almost always in mg.")
    weight_units_key = "mg" if weight_units == "mg" else "ug"

    f_cao = st.number_input(
        "Active CaO mass fraction (0–1)", min_value=0.01, max_value=1.0,
        value=1.0, step=0.01,
        help="Fraction of the sorbent's calcined mass that is active CaO. "
             "Use 1.0 for pure CaO; lower it for supported / composite sorbents "
             "(e.g. Ni/CaO dual-functional materials) if you know the CaO loading. "
             "This only affects the fractional-conversion X(t) used for kinetic "
             "fitting - the headline mmol CO2/g capacity is always on a whole-sorbent basis.",
    )

    dead_time_pct = st.slider(
        "Ignore pre-reaction dead time (%)", min_value=0, max_value=20, value=2, step=1,
        help="If your box-selection starts a little early, while the sample is "
             "still flat (not yet exposed to CO2), this drops everything before "
             "the first point where conversion crosses this % of that cycle's "
             "own max conversion, and resets t=0 / w0 to that point. So the fit "
             "only ever sees the real carbonation curve. Set to 0 to use the "
             "selection exactly as drawn.",
    )
    dead_time_frac = dead_time_pct / 100.0

    st.markdown("---")
    with st.expander("Advanced: fitting speed / resolution"):
        max_fit_points = st.slider(
            "Max points used for fitting (per cycle)", min_value=60, max_value=1000,
            value=150, step=10,
            help="TGA cycles can have 1000+ points; fitting curves are smooth, so "
                 "an even subsample of this many points is used for the (relatively "
                 "expensive) two-stage breakpoint search and curve fitting. Figures "
                 "and capacity calculations always use the full-resolution data. "
                 "Raise this for a more exhaustive fit at the cost of speed.",
        )
        coarse_candidates = st.slider(
            "Coarse breakpoint candidates per model", min_value=40, max_value=300,
            value=60, step=10,
            help="Higher = more precise transition-point search, slower.",
        )
    st.markdown("---")
    st.caption(
        "Two-stage fit: for each kinetic model, the carbonation curve is split into "
        "a fast stage [0, t_b] and a slow stage (t_b, t_end], each re-fit with its "
        "own rate constant. t_b is optimized independently for every model to "
        "minimize the combined residual sum of squares."
    )


# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

tab_load, tab_kinetics, tab_compare, tab_about = st.tabs(
    ["1. Load & select cycles", "2. Kinetics (single cycle)", "3. Compare cycles", "About / model reference"]
)


# ==========================================================================
# TAB 1 — Load & select cycles
# ==========================================================================
with tab_load:
    st.header("Load TGA data")
    uploaded = st.file_uploader(
        "Upload one or more TGA CSV exports", type=["csv"], accept_multiple_files=True,
        help="Standard TA Instruments-style export with Time, (Unsubtracted) Weight, "
             "and Program Temperature columns.",
    )
    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state["dataframes"]:
                try:
                    st.session_state["dataframes"][f.name] = load_tga_csv(f)
                except Exception as e:
                    st.error(f"Could not parse {f.name}: {e}")

    fname, df = get_active_df()

    if df is None:
        st.info("Upload at least one CSV file to get started.")
    else:
        file_names = list(st.session_state["dataframes"].keys())
        fname = st.selectbox("Active file", file_names, index=file_names.index(fname))
        st.session_state["active_file"] = fname
        df = st.session_state["dataframes"][fname]

        st.caption(f"{len(df):,} data points, "
                  f"{df['Time_min'].min():.2f}–{df['Time_min'].max():.2f} min")

        # ---------------- Raw chart with box-select ----------------
        st.subheader("Select each cycle's carbonation window")
        st.caption(
            "Drag a box across the plot (box-select tool, default in the toolbar) "
            "to mark one cycle's carbonation segment, then confirm it below. "
            "You can fine-tune the start/end times with the number inputs before adding."
        )

        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=df["Time_min"], y=df["Weight_mg"], mode="lines", name="Weight (mg)",
            line=dict(color="#1B6CA8", width=1.3), yaxis="y1",
        ))
        if "ProgramTemp_C" in df.columns:
            fig.add_trace(go.Scattergl(
                x=df["Time_min"], y=df["ProgramTemp_C"], mode="lines",
                name="Program temperature (°C)",
                line=dict(color="#C1272D", width=1.0, dash="dot"), yaxis="y2", opacity=0.6,
            ))
        fig.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title="Time (min)"),
            yaxis=dict(title="Weight (mg)"),
            yaxis2=dict(title="Program temperature (°C)", overlaying="y", side="right", showgrid=False),
            dragmode="select",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        )

        event = st.plotly_chart(
            fig, use_container_width=True, on_select="rerun",
            selection_mode=("box",), key="raw_chart",
        )

        sel_t0, sel_t1 = None, None
        try:
            sel = event.selection if hasattr(event, "selection") else event.get("selection", {})
            boxes = sel.get("box") if isinstance(sel, dict) else getattr(sel, "box", None)
            if boxes:
                b0 = boxes[0]
                xr = b0.get("x") if isinstance(b0, dict) else getattr(b0, "x", None)
                if xr and len(xr) == 2:
                    sel_t0, sel_t1 = float(min(xr)), float(max(xr))
        except Exception:
            pass

        t_min, t_max = float(df["Time_min"].min()), float(df["Time_min"].max())
        # A fresh box-selection should overwrite the number inputs even though
        # they already have a key in session_state from a previous run.
        if sel_t0 is not None and st.session_state.get("_last_sel") != (sel_t0, sel_t1):
            st.session_state["sel_start"] = sel_t0
            st.session_state["sel_end"] = sel_t1
            st.session_state["_last_sel"] = (sel_t0, sel_t1)
        st.session_state.setdefault("sel_start", t_min)
        st.session_state.setdefault("sel_end", t_min)

        c1, c2, c3 = st.columns(3)
        with c1:
            t_start = st.number_input(
                "Selected start (min)", min_value=t_min, max_value=t_max,
                format="%.4f", key="sel_start",
            )
        with c2:
            t_end = st.number_input(
                "Selected end (min)", min_value=t_min, max_value=t_max,
                format="%.4f", key="sel_end",
            )
        with c3:
            default_label = f"Cycle {st.session_state['next_cycle_index']}"
            cyc_label = st.text_input("Cycle label", value=default_label, key="sel_label")

        if st.button("Add selection as a cycle", type="primary"):
            if t_end <= t_start:
                st.error("End time must be greater than start time.")
            else:
                cyc = Cycle(label=cyc_label, t_start=t_start, t_end=t_end,
                           source_file=fname, cycle_index=st.session_state["next_cycle_index"])
                st.session_state["cycles"].append(cyc)
                st.session_state["next_cycle_index"] += 1
                st.success(f"Added '{cyc_label}' ({t_start:.2f}–{t_end:.2f} min).")

        # ---------------- Optional auto-detect helper ----------------
        with st.expander("Optional: auto-detect candidate cycles from a temperature plateau"):
            if "ProgramTemp_C" not in df.columns:
                st.write("No ProgramTemp_C column found in this file.")
            else:
                ac1, ac2, ac3 = st.columns(3)
                with ac1:
                    target_temp = st.number_input("Carbonation set-point (°C)", value=675.0)
                with ac2:
                    tol = st.number_input("Tolerance (± °C)", value=5.0)
                with ac3:
                    min_dur = st.number_input("Minimum duration (min)", value=5.0)
                if st.button("Detect segments"):
                    segs = detect_plateau_segments(df, target_temp, tol, min_dur)
                    st.session_state["detected_segments"] = segs
                segs = st.session_state.get("detected_segments", [])
                if segs:
                    seg_df = pd.DataFrame(segs, columns=["t_start", "t_end"])
                    seg_df["duration_min"] = seg_df["t_end"] - seg_df["t_start"]
                    seg_df.insert(0, "import", True)
                    seg_df.insert(1, "label", [f"Cycle {i+1}" for i in range(len(seg_df))])
                    edited = st.data_editor(seg_df, key="seg_editor", hide_index=True)
                    if st.button("Import checked segments as cycles"):
                        n_added = 0
                        for _, row in edited.iterrows():
                            if row["import"]:
                                cyc = Cycle(label=row["label"], t_start=float(row["t_start"]),
                                           t_end=float(row["t_end"]), source_file=fname,
                                           cycle_index=st.session_state["next_cycle_index"])
                                st.session_state["cycles"].append(cyc)
                                st.session_state["next_cycle_index"] += 1
                                n_added += 1
                        st.success(f"Imported {n_added} cycle(s).")

        # ---------------- Defined cycles table ----------------
        st.subheader("Defined cycles")
        cycles = st.session_state["cycles"]
        if not cycles:
            st.write("No cycles defined yet.")
        else:
            rows = []
            for i, cyc in enumerate(cycles):
                try:
                    res = extract_cycle(df if cyc.source_file == fname else
                                        st.session_state["dataframes"].get(cyc.source_file, df),
                                        cyc, f_cao=f_cao, weight_units=weight_units_key,
                                        dead_time_frac=dead_time_frac)
                    rows.append({
                        "#": i, "Label": cyc.label, "Cycle index": cyc.cycle_index,
                        "File": cyc.source_file, "t_start (min)": round(cyc.t_start, 3),
                        "t_end (min)": round(cyc.t_end, 3),
                        "Duration (min)": round(cyc.t_end - cyc.t_start, 3),
                        "Onset trimmed (min)": round(res.onset_trim_min, 3),
                        "w0 (mg)": round(res.w0_mg, 5),
                        "Capacity (mmol CO2/g)": round(res.capacity_mmol_per_g, 4),
                        "X_max": round(float(res.X.max()), 4),
                    })
                except Exception as e:
                    rows.append({"#": i, "Label": cyc.label, "Cycle index": cyc.cycle_index,
                               "File": cyc.source_file, "t_start (min)": cyc.t_start,
                               "t_end (min)": cyc.t_end, "Duration (min)": None,
                               "Onset trimmed (min)": None,
                               "w0 (mg)": None, "Capacity (mmol CO2/g)": None, "X_max": None})
            cyc_table = pd.DataFrame(rows)
            st.dataframe(cyc_table, use_container_width=True, hide_index=True)

            del_col1, del_col2 = st.columns([3, 1])
            with del_col1:
                to_remove = st.selectbox(
                    "Remove a cycle", options=["—"] + [c.label for c in cycles],
                )
            with del_col2:
                if st.button("Remove") and to_remove != "—":
                    st.session_state["cycles"] = [c for c in cycles if c.label != to_remove]
                    st.session_state["results_cache"].pop(to_remove, None)
                    st.rerun()

            st.download_button(
                "Download cycle summary (CSV)", df_to_csv_bytes(cyc_table),
                file_name="cycle_summary.csv", mime="text/csv",
            )


# ==========================================================================
# TAB 2 — Kinetics (single cycle)
# ==========================================================================
with tab_kinetics:
    st.header("Two-stage kinetic model fitting")
    cycles = st.session_state["cycles"]
    if not cycles:
        st.info("Define at least one cycle in tab 1 first.")
    else:
        labels = [c.label for c in cycles]
        chosen_label = st.selectbox("Cycle to analyze", labels, key="kin_cycle_select")
        cyc = next(c for c in cycles if c.label == chosen_label)
        src_df = st.session_state["dataframes"].get(cyc.source_file)
        cyc_res = extract_cycle(src_df, cyc, f_cao=f_cao, weight_units=weight_units_key,
                                dead_time_frac=dead_time_frac)
        if cyc_res.onset_trim_min > 0:
            st.caption(
                f"Trimmed {cyc_res.onset_trim_min:.2f} min of flat baseline before the "
                "detected reaction onset (t=0 below is the onset, not the raw selection start)."
            )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("w0 (calcined mass)", f"{cyc_res.w0_mg:.4f} mg")
        m2.metric("Capacity", f"{cyc_res.capacity_mmol_per_g:.3f} mmol CO2/g")
        m3.metric("Duration", f"{cyc_res.t_rel[-1]:.2f} min")
        m4.metric("Max conversion X", f"{cyc_res.X.max():.3f}")

        run = st.button("Run kinetic analysis for this cycle", type="primary")
        cache = st.session_state["results_cache"]
        if run:
            t_fit, X_fit = downsample_series(cyc_res.t_rel, cyc_res.X, max_points=max_fit_points)
            with st.spinner(
                f"Fitting 11 kinetic models on {len(t_fit)} points with per-model "
                "optimized transition points..."
            ):
                results = {}
                for key in MODEL_ORDER:
                    results[key] = fit_two_stage(
                        t_fit, X_fit, key, coarse_candidates=coarse_candidates,
                    )
                cache[chosen_label] = results

        results = cache.get(chosen_label)
        if results is None:
            st.info("Click **Run kinetic analysis** to fit all models.")
        else:
            rows = []
            for key, r in results.items():
                model = MODELS[key]
                rows.append({
                    "Model": model.label,
                    "key": key,
                    "R2": r.r2 if r.success else np.nan,
                    "AIC": r.aic if r.success else np.nan,
                    "BIC": r.bic if r.success else np.nan,
                    "Transition t_b (min)": r.breakpoint_t,
                    "R2 stage 1 (fast)": r.r2_stage1,
                    "R2 stage 2 (slow)": r.r2_stage2,
                    "Stage 1 params": ", ".join(f"{k}={v:.4g}" for k, v in (r.stage1_params or {}).items()),
                    "Stage 2 params": ", ".join(f"{k}={v:.4g}" for k, v in (r.stage2_params or {}).items()),
                    "n params": r.n_params,
                    "Converged": r.success,
                })
            table = pd.DataFrame(rows).sort_values("AIC", ascending=True).reset_index(drop=True)
            best_key = table.iloc[0]["key"]

            st.subheader("Model comparison (sorted by AIC, best first)")
            display_table = table.drop(columns=["key"]).copy()
            display_table.loc[0, "Model"] = "★ " + display_table.loc[0, "Model"]
            st.dataframe(
                display_table.style.format({
                    "R2": "{:.4f}", "AIC": "{:.2f}", "BIC": "{:.2f}",
                    "Transition t_b (min)": "{:.3f}",
                    "R2 stage 1 (fast)": "{:.4f}", "R2 stage 2 (slow)": "{:.4f}",
                }),
                use_container_width=True, hide_index=True,
            )

            plot_key = st.selectbox(
                "Model to plot", options=list(results.keys()),
                index=list(results.keys()).index(best_key),
                format_func=lambda k: MODELS[k].label + ("  ★ best (lowest AIC)" if k == best_key else ""),
            )
            fig = plot_cycle_fit(cyc_res, results[plot_key], MODELS[plot_key].label,
                                model_key1=plot_key)
            st.pyplot(fig, use_container_width=False)

            dl1, dl2, dl3, dl4, dl5 = st.columns(5)
            with dl1:
                st.download_button("Table (CSV)", df_to_csv_bytes(table.drop(columns=["key"])),
                                  file_name=f"{chosen_label}_kinetics.csv", mime="text/csv")
            with dl2:
                st.download_button(
                    "Table (Excel)",
                    dfs_to_excel_bytes({"kinetics": table.drop(columns=["key"])}),
                    file_name=f"{chosen_label}_kinetics.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with dl3:
                st.download_button("Figure (PNG)", fig_to_bytes(fig, "png"),
                                  file_name=f"{chosen_label}_{plot_key}.png", mime="image/png")
            with dl4:
                st.download_button("Figure (PDF)", fig_to_bytes(fig, "pdf"),
                                  file_name=f"{chosen_label}_{plot_key}.pdf", mime="application/pdf")
            with dl5:
                st.download_button("Figure (SVG)", fig_to_bytes(fig, "svg"),
                                  file_name=f"{chosen_label}_{plot_key}.svg", mime="image/svg+xml")

            # ---------------- Custom mixed two-stage exploration ----------------
            with st.expander("Advanced: mix different models for the fast vs. slow stage"):
                st.caption(
                    "The table above uses the *same* model on both stages (its own "
                    "transition point). Here you can instead pick one model for the "
                    "fast (reaction-controlled) stage and a different one for the slow "
                    "(diffusion-controlled) stage."
                )
                mc1, mc2 = st.columns(2)
                with mc1:
                    stage1_key = st.selectbox("Fast-stage model", MODEL_ORDER,
                                              format_func=lambda k: MODELS[k].label, key="mix_s1")
                with mc2:
                    stage2_key = st.selectbox("Slow-stage model", MODEL_ORDER,
                                              index=MODEL_ORDER.index("ginstling_brounshtein"),
                                              format_func=lambda k: MODELS[k].label, key="mix_s2")
                if st.button("Fit this combination"):
                    t_fit, X_fit = downsample_series(cyc_res.t_rel, cyc_res.X, max_points=max_fit_points)
                    mixed = fit_two_stage_mixed(t_fit, X_fit, stage1_key, stage2_key,
                                                coarse_candidates=coarse_candidates)
                    st.session_state["mixed_results_cache"][chosen_label] = (mixed, stage1_key, stage2_key)
                cached_mixed = st.session_state["mixed_results_cache"].get(chosen_label)
                if cached_mixed is not None and cached_mixed[0].success:
                    mixed, mixed_s1, mixed_s2 = cached_mixed
                    cm1, cm2, cm3 = st.columns(3)
                    cm1.metric("R2 (combined)", f"{mixed.r2:.4f}")
                    cm2.metric("AIC", f"{mixed.aic:.2f}")
                    cm3.metric("BIC", f"{mixed.bic:.2f}")
                    label = f"{MODELS[mixed_s1].label} + {MODELS[mixed_s2].label}"
                    mfig = plot_cycle_fit(cyc_res, mixed, label, model_key1=mixed_s1, model_key2=mixed_s2)
                    st.pyplot(mfig, use_container_width=False)
                    st.download_button("Mixed-fit figure (PNG)", fig_to_bytes(mfig, "png"),
                                      file_name=f"{chosen_label}_mixed.png", mime="image/png")


# ==========================================================================
# TAB 3 — Compare cycles
# ==========================================================================
with tab_compare:
    st.header("Compare cycles")
    cycles = st.session_state["cycles"]
    if not cycles:
        st.info("Define at least one cycle in tab 1 first.")
    else:
        labels = [c.label for c in cycles]
        chosen = st.multiselect("Cycles to compare", labels, default=labels)
        selected_cycles = [c for c in cycles if c.label in chosen]

        if selected_cycles:
            cycle_results = []
            for c in selected_cycles:
                src_df = st.session_state["dataframes"].get(c.source_file)
                cycle_results.append(extract_cycle(src_df, c, f_cao=f_cao, weight_units=weight_units_key,
                                                   dead_time_frac=dead_time_frac))

            summary_rows = [{
                "Label": cr.label, "Cycle index": cr.cycle_index,
                "Capacity (mmol CO2/g)": cr.capacity_mmol_per_g,
                "w0 (mg)": cr.w0_mg, "X_max": float(cr.X.max()),
                "Duration (min)": float(cr.t_rel[-1]),
            } for cr in cycle_results]
            # attach best-fit model info if kinetics were already run for that cycle
            for row in summary_rows:
                res = st.session_state["results_cache"].get(row["Label"])
                if res:
                    best_key = min(res, key=lambda k: res[k].aic if res[k].success else np.inf)
                    row["Best model (lowest AIC)"] = MODELS[best_key].label
                    row["Best model R2"] = res[best_key].r2
                    row["Best model AIC"] = res[best_key].aic
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

            cfig1, cfig2 = st.columns(2)
            with cfig1:
                bar_fig = plot_capacity_bar(
                    [cr.label for cr in cycle_results],
                    [cr.capacity_mmol_per_g for cr in cycle_results],
                )
                st.pyplot(bar_fig, use_container_width=False)
                st.download_button("Capacity chart (PNG)", fig_to_bytes(bar_fig, "png"),
                                  file_name="capacity_comparison.png", mime="image/png")
            with cfig2:
                curves_fig = plot_cycle_comparison_curves(cycle_results)
                st.pyplot(curves_fig, use_container_width=False)
                st.download_button("Curves chart (PNG)", fig_to_bytes(curves_fig, "png"),
                                  file_name="curves_comparison.png", mime="image/png")

            st.download_button(
                "Download comparison summary (Excel)",
                dfs_to_excel_bytes({
                    "summary": summary_df,
                    **{cr.label: pd.DataFrame({"t_min": cr.t_rel, "X": cr.X,
                                              "weight_mg": cr.weight_mg}) for cr in cycle_results},
                }),
                file_name="cycle_comparison.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.subheader("Optional: cycle-to-cycle capacity decay (Grasa–Abanades model)")
            st.caption(
                "Fits X_N = X_r + 1 / (1/(1-X_r) + k·N) to capacity vs. cycle number — "
                "the standard model for describing the loss of CO2 capture capacity over "
                "repeated calcination/carbonation cycles."
            )
            if len(selected_cycles) >= 3:
                Ns = [cr.cycle_index for cr in cycle_results]
                caps = [cr.capacity_mmol_per_g for cr in cycle_results]
                fit = fit_grasa_abanades(Ns, caps)
                if fit:
                    N_smooth = np.linspace(min(Ns), max(Ns), 200)
                    cap_smooth = f_grasa_abanades(N_smooth, fit["Xr"], fit["k"])
                    deac_fig = plot_deactivation_fit(Ns, caps, (N_smooth, cap_smooth), fit)
                    dcol1, dcol2 = st.columns([1, 2])
                    with dcol1:
                        st.metric("Residual capacity, X_r", f"{fit['Xr']:.3f}")
                        st.metric("Deactivation constant, k", f"{fit['k']:.4f}")
                        st.metric("R2", f"{fit['r2']:.4f}")
                    with dcol2:
                        st.pyplot(deac_fig, use_container_width=False)
                    st.download_button("Decay fit figure (PNG)", fig_to_bytes(deac_fig, "png"),
                                      file_name="capacity_decay_fit.png", mime="image/png")
                else:
                    st.warning("Could not fit the deactivation model to these cycles.")
            else:
                st.write("Select at least 3 cycles (with distinct cycle indices) to fit the decay model.")


# ==========================================================================
# ABOUT / MODEL REFERENCE
# ==========================================================================
with tab_about:
    st.header("Method reference")

    st.markdown("""
**Capture capacity.** For each defined cycle, `w0` is the sample mass at the start of
the selected carbonation window (the calcined mass) and the CO2 capture capacity is
computed directly from the mass gain:

`capacity (mmol CO2/g) = (w(t_end) - w0) / M_CO2 * 1000 / w0`, with M_CO2 = 44.01 g/mol.

**Fractional conversion for kinetics.** The kinetic models below are fit to the
fractional conversion of CaO, normalized by the theoretical maximum mass gain for
the sorbent's active CaO content:

`X(t) = (w(t) - w0) / (w0 * f_CaO * (M_CO2 / M_CaO))`, with M_CaO = 56.08 g/mol.

`f_CaO` (set in the sidebar) is the active CaO mass fraction of the sorbent — use
1.0 for pure CaO and a lower value for supported/composite sorbents such as Ni/CaO
dual-functional materials.

**Two-stage fitting.** Calcium-looping carbonation curves typically show a fast,
chemically/reaction-controlled regime followed by a slower regime limited by CO2
diffusion through the growing CaCO3 product layer. For each kinetic model, the app
searches over possible transition times t_b and, for each candidate, independently
fits the model to the data before t_b (fast stage) and to the *remaining* conversion
after t_b (slow stage, renormalized so the same functional form can be reused with
its own rate constant), then keeps the t_b that minimizes the combined residual sum
of squares. AIC/BIC for the two-stage fit count all fitted parameters from both
stages plus the transition point itself.
""")

    st.subheader("Kinetic models included")
    ref_rows = []
    descriptions = {
        "zero_order": "Reaction control, planar/linear geometry: X = kt.",
        "first_order": "Apparent first-order (most widely used fast-stage model): X = 1 - exp(-kt).",
        "second_order": "Apparent second-order: X/(1-X) = kt.",
        "nth_order": "Apparent n-th order, with n fitted alongside k.",
        "avrami_erofeev": "Avrami-Erofeev / JMAK nucleation-and-growth model: X = 1 - exp(-(kt)^n).",
        "scm_sphere": "Shrinking Core Model, reaction control, spherical particle (contracting volume).",
        "scm_cylinder": "Shrinking Core Model, reaction control, cylindrical particle (contracting area).",
        "jander": "Jander diffusion model: product-layer diffusion control, spherical geometry.",
        "ginstling_brounshtein": "Ginstling-Brounshtein diffusion model: product-layer diffusion control, spherical geometry (solved numerically).",
        "parabolic": "Parabolic diffusion law: 1-D product-layer diffusion control.",
        "random_pore_model": "Bhatia-Perlmutter Random Pore Model, accounting for pore structure evolution via the structural parameter psi.",
    }
    for key in MODEL_ORDER:
        m = MODELS[key]
        ref_rows.append({"Model": m.label, "Type": m.stage_type,
                        "Parameters": ", ".join(m.param_names),
                        "Description": descriptions.get(key, "")})
    st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)

    st.subheader("Scoring")
    st.markdown("""
- **R2** — coefficient of determination of the fit against the observed conversion curve.
- **AIC** = n·ln(RSS/n) + 2k
- **BIC** = n·ln(RSS/n) + k·ln(n)

where n is the number of data points, RSS the residual sum of squares, and k the
number of fitted parameters (for a two-stage fit: both stages' parameters plus one
for the fitted transition point). Lower AIC/BIC indicates a better trade-off between
fit quality and model complexity — use it, not R2 alone, to compare models with a
different number of parameters.
""")
