"""IMMUNOVISION_X – immunodiffusion analysis and teaching workspace."""
import io
import json
import tempfile
from pathlib import Path

import altair as alt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import science as sc
from analysis_engine import DID_PATTERNS, analyse_did, analyse_srid
from content import GLOSSARY, QUIZ, THEORY
from demo import DEMO_CONCENTRATIONS, make_did_demo, make_srid_demo

st.set_page_config(page_title="IMMUNOVISION_X", page_icon=None,
                   layout="wide", initial_sidebar_state="expanded")

NAVY, TEAL, AMBER, SLATE, RED = "#1b3a57", "#2a7f8e", "#c98a1b", "#6b7c8d", "#b5473b"

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
  h1, h2, h3, h4 {{ color: {NAVY}; letter-spacing: -0.01em; }}
  .brand {{ font-size: 1.55rem; font-weight: 700; color: {NAVY};
            letter-spacing: .02em; margin: 0; }}
  .brand span {{ color: {TEAL}; }}
  .tagline {{ color: {SLATE}; font-size: .92rem; margin-bottom: .2rem; }}
  .pagehead {{ border-bottom: 2px solid {NAVY}; padding-bottom: .5rem;
               margin-bottom: 1.1rem; }}
  .pagehead h2 {{ margin: 0; font-size: 1.45rem; }}
  .pagehead p {{ margin: .2rem 0 0; color: {SLATE}; font-size: .95rem; }}
  div[data-testid="stMetric"] {{ background: #fff; border: 1px solid #d9e0e7;
        border-left: 4px solid {TEAL}; border-radius: 4px; padding: 12px 14px; }}
  div[data-testid="stMetricLabel"] p {{ color: {SLATE}; font-size: .8rem;
        text-transform: uppercase; letter-spacing: .05em; }}
  .card {{ background: #fff; border: 1px solid #d9e0e7; border-radius: 4px;
           padding: 16px 18px; height: 100%; }}
  .card h4 {{ margin: 0 0 .4rem; font-size: 1.02rem; }}
  .card p {{ margin: 0; color: #3b4b5a; font-size: .92rem; line-height: 1.5; }}
  .step {{ font-weight: 700; color: {TEAL}; font-size: .8rem;
           letter-spacing: .08em; }}
  .note {{ background: #f1f5f8; border-left: 3px solid {SLATE};
           padding: 10px 14px; font-size: .9rem; color: #33475a; }}
  section[data-testid="stSidebar"] {{ border-right: 1px solid #d9e0e7; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------
DEFAULTS = {
    "srid": None,            # dict: report, input_path, table
    "did": None,
    "calibration": None,     # science.Calibration
    "srid_results": None,
    "quiz_answers": {},
    "quiz_submitted": False,
    "quiz_nonce": 0,
}
for key, val in DEFAULTS.items():
    st.session_state.setdefault(key, val)


def head(title, subtitle):
    st.markdown(f'<div class="pagehead"><h2>{title}</h2><p>{subtitle}</p>'
                '</div>', unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


def card(title, body, step=None):
    s = f'<div class="step">{step}</div>' if step else ""
    st.markdown(f'<div class="card">{s}<h4>{title}</h4><p>{body}</p></div>',
                unsafe_allow_html=True)


def workdir(prefix):
    return Path(tempfile.mkdtemp(prefix=prefix))


def save_upload(uploaded, folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / Path(uploaded.name).name
    path.write_bytes(uploaded.getvalue())
    return path


def fmt(value, digits=3):
    return "–" if value is None or (isinstance(value, float)
                                    and np.isnan(value)) else f"{value:.{digits}f}"


# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
PAGES = ["Overview", "SRID Analysis", "Calibration", "DID Analysis",
         "Diffusion Lab", "Learn", "Quiz", "Reports & Export"]
with st.sidebar:
    st.markdown('<p class="brand">IMMUNOVISION<span>_X</span></p>',
                unsafe_allow_html=True)
    st.markdown('<p class="tagline">Immunodiffusion analysis workspace</p>',
                unsafe_allow_html=True)
    st.divider()
    page = st.radio("Workspace", PAGES, label_visibility="collapsed")
    st.divider()
    s = st.session_state
    st.caption("Session status")
    st.write(f"SRID plate: {'loaded' if s.srid else 'none'}")
    st.write(f"Calibration: {'fitted' if s.calibration else 'none'}")
    st.write(f"DID plate: {'loaded' if s.did else 'none'}")
    st.divider()
    st.caption("Automatic detections are candidates. Verify every ring "
               "and line against the physical plate.")


# ----------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------
def bar_chart(df, x, y, title, color=TEAL):
    return (alt.Chart(df).mark_bar(color=color, size=26)
            .encode(x=alt.X(x, title=None, sort=None),
                    y=alt.Y(y, title=title),
                    tooltip=list(df.columns))
            .properties(height=260))


def pie_chart(df, cat, val, colors=None):
    scale = alt.Scale(range=colors) if colors else alt.Undefined
    return (alt.Chart(df).mark_arc(innerRadius=55, outerRadius=100)
            .encode(theta=alt.Theta(val, type="quantitative"),
                    color=alt.Color(cat, type="nominal", scale=scale,
                                    legend=alt.Legend(title=None)),
                    tooltip=[cat, val])
            .properties(height=250))


def calibration_chart(cal, stds, unknowns=None):
    cs, ys = sc.calibration_curve_points(cal)
    line = pd.DataFrame({"conc": cs, "y": ys})
    ylab = "D²" if cal.method == "mancini" else "Ring diameter"
    base_line = alt.Chart(line).mark_line(color=NAVY).encode(
        x=alt.X("conc", title=f"Concentration ({cal.unit_conc})"),
        y=alt.Y("y", title=f"{ylab} ({cal.unit_diameter}"
                           f"{'²' if cal.method == 'mancini' else ''})"))
    pts = alt.Chart(stds).mark_point(color=TEAL, filled=True, size=90).encode(
        x="conc", y="y", tooltip=["conc", "y"])
    layers = [base_line, pts]
    if unknowns is not None and len(unknowns):
        layers.append(alt.Chart(unknowns).mark_point(
            color=AMBER, shape="diamond", filled=True, size=130).encode(
            x="conc", y="y", tooltip=["conc", "y"]))
    return alt.layer(*layers).properties(height=320)


# ----------------------------------------------------------------------
# DID schematics (teaching)
# ----------------------------------------------------------------------
def did_schematic(kind):
    fig, ax = plt.subplots(figsize=(3.2, 2.6), dpi=130)
    ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis("off")
    for (x, y), label in [((5, 6.7), "Ab"), ((2.2, 1.6), "Ag 1"),
                          ((7.8, 1.6), "Ag 2")]:
        ax.add_patch(plt.Circle((x, y), .6, fc="#dfe6ec", ec=NAVY, lw=1.4))
        ax.text(x, y, label, ha="center", va="center", fontsize=7,
                color=NAVY)
    if kind == "identity":
        ax.plot([2.4, 3.6, 5, 6.4, 7.6], [3.2, 4.0, 4.3, 4.0, 3.2],
                color=TEAL, lw=2.4, solid_capstyle="round")
    elif kind == "partial_identity":
        ax.plot([2.4, 3.6, 5, 6.4, 7.6], [3.2, 4.0, 4.3, 4.0, 3.2],
                color=TEAL, lw=2.4)
        ax.plot([5, 5.9, 6.7], [4.3, 3.0, 2.2], color=AMBER, lw=2)
    elif kind == "non_identity":
        ax.plot([2.2, 4.2, 6.3], [2.4, 3.6, 5.1], color=TEAL, lw=2.4)
        ax.plot([7.8, 5.8, 3.7], [2.4, 3.6, 5.1], color=AMBER, lw=2.4)
    ax.set_title(kind.replace("_", " ").title(), fontsize=9, color=NAVY)
    plt.close(fig)
    return fig


# ======================================================================
# OVERVIEW
# ======================================================================
if page == "Overview":
    head("Overview", "Quantitative and qualitative analysis of "
         "antigen–antibody precipitation by SRID and DID.")

    srid = st.session_state.srid
    res = st.session_state.srid_results
    cal = st.session_state.calibration
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Wells detected", srid["report"]["total_well_candidates"]
              if srid else "–")
    m2.metric("Rings detected", srid["report"]["ring_candidates"]
              if srid else "–")
    m3.metric("Calibration R²", fmt(cal.r2, 4) if cal else "–")
    unk = 0
    if res is not None:
        unk = int(res["estimated_conc"].notna().sum())
    m4.metric("Concentrations estimated", unk if res is not None else "–")

    st.write("")
    st.markdown("#### Workflow")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("Load a plate", "Upload an SRID photo or start with the "
             "built-in teaching plate.", "STEP 1")
    with c2:
        card("Verify rings", "Check each detected ring and overwrite the "
             "diameter where the software is wrong.", "STEP 2")
    with c3:
        card("Calibrate", "Assign standards. A line of D² against "
             "concentration is fitted automatically.", "STEP 3")
    with c4:
        card("Report", "Concentrations, charts and CSV/JSON export with "
             "full provenance.", "STEP 4")

    st.write("")
    a, b = st.columns(2)
    with a:
        st.markdown("#### SRID – quantification")
        st.write("Antigen diffuses from a well into antibody-containing "
                 "agarose. The ring diameter (squared) is proportional to "
                 "antigen concentration at end-point, which lets unknown "
                 "samples be read from standards.")
    with b:
        st.markdown("#### DID – identity testing")
        st.write("Antigen and antibody diffuse towards each other and "
                 "precipitate where they meet. The geometry of the lines "
                 "shows identity, partial identity or non-identity.")

    if res is not None:
        st.markdown("#### Current plate")
        unit = "mm" if "D2_mm2" in res.columns else "px"
        d2col = f"D2_{unit}2"
        ch1, ch2 = st.columns([3, 2])
        with ch1:
            chart_df = res.assign(well=res["well"].astype(str))
            st.altair_chart(bar_chart(chart_df, "well", d2col,
                                      f"D² ({unit}²)"))
        with ch2:
            roles = res["role"].value_counts().reset_index()
            roles.columns = ["role", "wells"]
            st.altair_chart(pie_chart(roles, "role", "wells",
                                      [TEAL, AMBER, SLATE]))
    else:
        note("No plate analysed yet. Open <b>SRID Analysis</b> and choose "
             "<i>Use teaching plate</i> to see the complete workflow in "
             "seconds.")


# ======================================================================
# SRID
# ======================================================================
elif page == "SRID Analysis":
    head("SRID Analysis", "Detect wells and rings, verify them, calibrate "
         "and estimate concentration.")

    src_col, set_col = st.columns([3, 2])
    with src_col:
        uploaded = st.file_uploader("SRID plate image",
                                    type=["png", "jpg", "jpeg", "tif", "tiff"])
        use_demo = st.checkbox("Use teaching plate (synthetic data)")
    with set_col:
        with st.expander("Detection settings", expanded=False):
            auto_r = st.checkbox("Automatic minimum well radius", True)
            min_r = None if auto_r else st.number_input(
                "Minimum well radius (px)", 4, 200, 12)
            thr = st.slider("Ring support threshold", 0.10, 0.80, 0.28, 0.01,
                            help="Fraction of the circle that must lie on "
                                 "an edge. Lower finds fainter rings but "
                                 "adds false positives.")
            three = st.checkbox("Image is a 3-panel A/B/C figure")
        known_mm = st.number_input(
            "Known well diameter (mm) – optional", 0.0, 50.0, 3.0 if use_demo
            else 0.0, 0.1, help="Sets the pixel-to-mm scale. Leave at 0 to "
                                "work in pixels.")

    if st.button("Analyse plate", type="primary"):
        run = workdir("ivx_srid_")
        try:
            if use_demo:
                path = make_srid_demo(run / "teaching_plate.png")
            elif uploaded:
                path = save_upload(uploaded, run / "input")
            else:
                st.warning("Upload an image or tick 'Use teaching plate'.")
                st.stop()
            with st.spinner("Detecting wells and rings..."):
                report = analyse_srid(path, run / "out", min_r, thr,
                                      "three" if three else "single")
            meas = pd.DataFrame(report["measurements"])
            if meas.empty:
                st.session_state.srid = None
                st.error("No wells were detected. Try lowering the minimum "
                         "well radius or ring threshold, or upload a "
                         "sharper, evenly lit image.")
                st.image(str(report["annotated_image"]))
                st.stop()
            roles = ["Unknown"] * len(meas)
            conc = [np.nan] * len(meas)
            if use_demo and len(meas) == 6:
                roles = ["Standard"] * 5 + ["Unknown"]
                conc = DEMO_CONCENTRATIONS[:5] + [np.nan]
            table = pd.DataFrame({
                "well_number": meas["well_number"].astype(int),
                "role": roles,
                "standard_conc": conc,
                "candidate_diameter_px": meas["candidate_ring_diameter_px"],
                "verified_diameter_px": np.nan,
            })
            st.session_state.srid = {
                "report": report, "table": table, "demo": use_demo,
                "input_path": str(path), "known_mm": known_mm,
            }
            st.session_state.pop("srid_editor", None)
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    S = st.session_state.srid
    if S is None:
        note("Upload a plate image and press <b>Analyse plate</b>.")
        st.stop()

    report, table = S["report"], S["table"]
    meas = pd.DataFrame(report["measurements"])
    wd_median = float(meas["well_diameter_px"].median())

    if S["demo"]:
        note("Teaching plate: synthetic image. Wells 1–5 are pre-assigned "
             "as standards (0.5–2.5 mg/mL); well 6 is an unknown whose true "
             "value is 1.75 mg/mL.")
    st.write("")

    tab_img, tab_meas, tab_conc, tab_charts = st.tabs(
        ["Plate", "Verify measurements", "Concentration", "Charts"])

    with tab_img:
        c1, c2 = st.columns(2)
        c1.image(S["input_path"], caption="Original")
        c2.image(report["annotated_image"],
                 caption="Detected wells (pink, numbered) and rings (orange)")
        a, b, c = st.columns(3)
        a.metric("Wells", report["total_well_candidates"])
        b.metric("Rings", report["ring_candidates"])
        c.metric("Median well diameter", f"{wd_median:.0f} px")
        if report["ring_candidates"] < report["total_well_candidates"]:
            note("Some wells have no ring candidate. Enter a diameter "
                 "manually in the next tab if you can see a ring.")

    with tab_meas:
        st.caption("Set each well's role. Enter a verified diameter "
                   "(pixels in the analysed image) only after inspecting "
                   "the ring; it overrides the automatic candidate.")
        edited = st.data_editor(
            table, key="srid_editor", hide_index=True, num_rows="fixed",
            column_config={
                "well_number": st.column_config.NumberColumn(
                    "Well", disabled=True),
                "role": st.column_config.SelectboxColumn(
                    "Role", options=["Standard", "Unknown", "Ignore"],
                    required=True),
                "standard_conc": st.column_config.NumberColumn(
                    "Standard conc.", min_value=0.0, format="%.4g"),
                "candidate_diameter_px": st.column_config.NumberColumn(
                    "Auto diameter (px)", disabled=True, format="%.0f"),
                "verified_diameter_px": st.column_config.NumberColumn(
                    "Verified diameter (px)", min_value=0.0, format="%.1f"),
            })

    with tab_conc:
        o1, o2, o3, o4 = st.columns(4)
        method = o1.selectbox("Method", list(sc.METHODS),
                              format_func=sc.METHODS.get)
        unit_conc = o2.text_input("Concentration unit", "mg/mL")
        dilution = o3.number_input("Sample dilution factor", 1.0, 1e6, 1.0)
        subtract = o4.checkbox("Subtract well D²", False,
                               disabled=method != "mancini",
                               help="Corrects D² for the well's own area.")

        results, cal, notes = sc.build_srid_results(
            edited, wd_median, known_mm if known_mm > 0 else None,
            method, subtract, dilution, unit_conc)
        st.session_state.srid_results = results
        st.session_state.calibration = cal

        unit = "mm" if known_mm > 0 else "px"
        if cal is not None:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("R²", fmt(cal.r2, 4))
            k2.metric("Quality", cal.quality())
            k3.metric("Standards used", cal.n)
            k4.metric("Range", f"{cal.conc_min:g}–{cal.conc_max:g} "
                               f"{unit_conc}")
            st.code(cal.equation, language=None)
        for n in notes:
            st.warning(n)

        show = results.copy()
        st.dataframe(show, hide_index=True, column_config={
            "estimated_conc": st.column_config.NumberColumn(
                f"Estimated conc. ({unit_conc})", format="%.4g"),
            "standard_conc": st.column_config.NumberColumn(
                f"Standard conc.", format="%.4g"),
        })
        st.caption("Concentration appears for wells marked 'Unknown' once "
                   "three or more standards have a concentration and a "
                   "ring diameter. 'extrapolated' means the ring lies "
                   "outside the standards' range.")
        st.download_button("Download results (CSV)",
                           results.to_csv(index=False).encode(),
                           "IMMUNOVISION_X_SRID_results.csv", "text/csv")

    with tab_charts:
        unit = "mm" if known_mm > 0 else "px"
        d2col = f"D2_{unit}2"
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Ring size by well")
            st.altair_chart(bar_chart(
                results.assign(well=results["well"].astype(str)),
                "well", d2col, f"D² ({unit}²)"))
        with c2:
            st.markdown("##### Well roles")
            roles = results["role"].value_counts().reset_index()
            roles.columns = ["role", "wells"]
            st.altair_chart(pie_chart(roles, "role", "wells",
                                      [TEAL, AMBER, SLATE]))
        if cal is not None:
            st.markdown("##### Calibration curve")
            sub = results[results["role"] == "Standard"].dropna(
                subset=["standard_conc", f"diameter_{unit}"])
            d = sub[f"diameter_{unit}"].to_numpy()
            y = d ** 2 - cal.well_diameter ** 2 if cal.method == "mancini" \
                else d
            stds = pd.DataFrame({"conc": sub["standard_conc"].to_numpy(),
                                 "y": y})
            u = results[results["estimated_conc"].notna()]
            du = u[f"diameter_{unit}"].to_numpy()
            yu = du ** 2 - cal.well_diameter ** 2 if cal.method == "mancini" \
                else du
            unk = pd.DataFrame({"conc": u["estimated_conc"].to_numpy()
                                / dilution, "y": yu})
            st.altair_chart(calibration_chart(cal, stds, unk))
            st.caption("Teal = standards, amber diamonds = unknowns, "
                       "navy = fitted line.")
        else:
            note("Charts of the calibration line appear once at least three "
                 "standards are defined.")


# ======================================================================
# CALIBRATION (manual)
# ======================================================================
elif page == "Calibration":
    head("Calibration", "Fit a standard curve from measured values and "
         "convert ring diameters to concentration.")
    note("Use this page when you measured standards by hand (for example "
         "with a ruler or calipers). Plates analysed on the SRID page "
         "calibrate automatically.")
    st.write("")
    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Method", list(sc.METHODS),
                          format_func=sc.METHODS.get)
    d_unit = c2.selectbox("Diameter unit", ["mm", "px"])
    c_unit = c3.text_input("Concentration unit", "mg/mL")

    start = pd.DataFrame({
        "concentration": [0.5, 1.0, 1.5, 2.0, 2.5],
        "diameter": [np.nan] * 5,
    })
    stds = st.data_editor(
        start, num_rows="dynamic", key="manual_cal",
        column_config={
            "concentration": st.column_config.NumberColumn(
                f"Concentration ({c_unit})", min_value=0.0, format="%.4g"),
            "diameter": st.column_config.NumberColumn(
                f"Ring diameter ({d_unit})", min_value=0.0, format="%.3f"),
        })
    well_d = st.number_input(f"Well diameter ({d_unit}) – used only for "
                             "Mancini well correction (0 = none)", 0.0, 100.0,
                             0.0)
    if st.button("Fit calibration", type="primary"):
        try:
            cal = sc.fit_calibration(stds["concentration"], stds["diameter"],
                                     method, well_d, d_unit, c_unit)
            st.session_state.calibration = cal
        except ValueError as exc:
            st.error(str(exc))

    cal = st.session_state.calibration
    if cal is not None:
        st.markdown("#### Fitted curve")
        k1, k2, k3 = st.columns(3)
        k1.metric("R²", fmt(cal.r2, 4))
        k2.metric("Quality", cal.quality())
        k3.metric("Standards", cal.n)
        st.code(cal.equation, language=None)

        d = np.asarray(stds["diameter"], float)
        c = np.asarray(stds["concentration"], float)
        ok = np.isfinite(d) & np.isfinite(c) & (d > 0) & (c > 0)
        y = d[ok] ** 2 - cal.well_diameter ** 2 if cal.method == "mancini" \
            else d[ok]
        pts = pd.DataFrame({"conc": c[ok], "y": y})
        st.altair_chart(calibration_chart(cal, pts))

        st.markdown("#### Estimate unknowns")
        q = st.data_editor(pd.DataFrame({"diameter": [np.nan] * 3}),
                           num_rows="dynamic", key="manual_unknown")
        dil = st.number_input("Dilution factor", 1.0, 1e6, 1.0,
                              key="manual_dil")
        if q["diameter"].notna().any():
            conc, flags = sc.predict_concentration(
                cal, q["diameter"].to_numpy(float), dil)
            st.dataframe(pd.DataFrame({
                f"diameter ({cal.unit_diameter})": q["diameter"],
                f"concentration ({cal.unit_conc})": conc,
                "flag": flags}), hide_index=True)


# ======================================================================
# DID
# ======================================================================
elif page == "DID Analysis":
    head("DID Analysis", "Locate wells and candidate precipitin lines; "
         "record your interpretation of the pattern.")
    c1, c2 = st.columns([3, 2])
    with c1:
        uploaded = st.file_uploader("DID plate image",
                                    type=["png", "jpg", "jpeg", "tif", "tiff"],
                                    key="did_upload")
        use_demo = st.checkbox("Use teaching plate (synthetic data)",
                               key="did_demo")
    with c2:
        interp = st.selectbox("Your interpretation", list(DID_PATTERNS),
                              format_func=lambda k: k.replace("_", " ")
                              .capitalize())
        comment = st.text_area("Observation notes", height=90)

    if st.button("Analyse DID plate", type="primary"):
        run = workdir("ivx_did_")
        try:
            if use_demo:
                path = make_did_demo(run / "teaching_did.png")
            elif uploaded:
                path = save_upload(uploaded, run / "input")
            else:
                st.warning("Upload an image or tick 'Use teaching plate'.")
                st.stop()
            with st.spinner("Processing..."):
                report = analyse_did(path, run / "out", interp)
            st.session_state.did = {"report": report, "path": str(path),
                                    "interp": interp, "comment": comment}
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.stop()

    D = st.session_state.did
    if D:
        rep = D["report"]
        a, b, c = st.columns(3)
        a.metric("Wells", rep["well_candidates"])
        b.metric("Line candidates", rep["line_candidates"])
        c.metric("Interpretation",
                 D["interp"].replace("_", " ").capitalize())
        i1, i2 = st.columns(2)
        i1.image(D["path"], caption="Original")
        i2.image(rep["annotated_image"],
                 caption="Wells (pink) and candidate lines (green)")
        if D["comment"]:
            st.markdown(f"**Notes:** {D['comment']}")
        st.markdown("**Interpretation guide:** "
                    + DID_PATTERNS[D["interp"]])
        t1, t2 = st.tabs(["Wells", "Centre-to-centre distances"])
        with t1:
            st.dataframe(pd.DataFrame(rep["wells"]), hide_index=True)
        with t2:
            dist = pd.DataFrame(rep["well_distances"])
            st.dataframe(dist, hide_index=True)
            st.caption("Distances in pixels. Divide by your plate's "
                       "px/mm scale for physical spacing, which feeds "
                       "the Diffusion Lab.")
        note(rep["notice"])
    else:
        note("Analyse a plate to see detected wells and lines.")

    st.markdown("#### Reference patterns")
    cols = st.columns(3)
    for col, kind in zip(cols, ["identity", "partial_identity",
                                "non_identity"]):
        with col:
            st.pyplot(did_schematic(kind))
            st.caption(DID_PATTERNS[kind])


# ======================================================================
# DIFFUSION LAB
# ======================================================================
elif page == "Diffusion Lab":
    head("Diffusion Lab", "Explore Fick's laws and how diffusion controls "
         "ring and line formation.")
    tab1, tab2, tab3 = st.tabs(["Concentration profile (Fick II)",
                                "Ring growth in SRID",
                                "Stokes–Einstein"])
    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            ref = st.selectbox("Protein", list(sc.DIFFUSION_REFERENCE) +
                               ["Custom"])
            D_cm = (st.number_input("D (cm²/s)", 1e-9, 1e-4, 4.0e-7,
                                    format="%.2e")
                    if ref == "Custom" else sc.DIFFUSION_REFERENCE[ref])
            gel = st.slider("Gel hindrance factor", 0.2, 1.0, 0.6, 0.05,
                            help="Agarose slows diffusion relative to free "
                                 "solution.")
            hours = st.slider("Time (hours)", 1, 72, 24)
            C0 = st.number_input("Source concentration C₀", 0.1, 100.0, 1.0)
            D_eff = D_cm * gel
            x_len = sc.diffusion_length(D_eff, hours * 3600)
            st.metric("Diffusion distance 2√(Dt)", f"{x_len * 10:.1f} mm")
            st.metric("Effective D", f"{D_eff:.2e} cm²/s")
        with c2:
            xs = np.linspace(0, max(x_len * 2.5, 0.2), 120)
            frames = []
            for frac in (0.1, 0.25, 0.5, 1.0):
                t = hours * 3600 * frac
                frames.append(pd.DataFrame({
                    "x_mm": xs * 10,
                    "C": sc.fick_second_profile(xs, t, D_eff, C0),
                    "time": f"{hours * frac:.1f} h"}))
            prof = pd.concat(frames)
            st.altair_chart(alt.Chart(prof).mark_line(strokeWidth=2.2)
                            .encode(x=alt.X("x_mm", title="Distance (mm)"),
                                    y=alt.Y("C", title="C / C₀ × C₀"),
                                    color=alt.Color("time", legend=alt.Legend(
                                        title="Time")),
                                    tooltip=["time", "x_mm", "C"])
                            .properties(height=340))
            st.latex(r"C(x,t)=C_0\,\mathrm{erfc}\!\left(\frac{x}{2\sqrt{Dt}}\right)")
            st.latex(r"J=-D\,\frac{\partial C}{\partial x}")
    with tab2:
        st.write("In SRID the ring radius grows roughly with √t until "
                 "end-point. Compare two concentrations.")
        k = st.slider("Growth constant (mm/√h)", 0.2, 3.0, 1.0, 0.1)
        end_r = st.slider("End-point radius (mm)", 2.0, 15.0, 6.0, 0.5)
        th = np.linspace(0, 72, 150)
        r1 = np.minimum(k * np.sqrt(th), end_r)
        r2 = np.minimum(1.4 * k * np.sqrt(th), end_r * 1.25)
        df = pd.concat([
            pd.DataFrame({"h": th, "radius": r1, "sample": "Lower conc."}),
            pd.DataFrame({"h": th, "radius": r2, "sample": "Higher conc."})])
        st.altair_chart(alt.Chart(df).mark_line(strokeWidth=2.2).encode(
            x=alt.X("h", title="Time (h)"),
            y=alt.Y("radius", title="Ring radius (mm)"),
            color=alt.Color("sample", legend=alt.Legend(title=None)))
            .properties(height=320))
        note("Plateau = end-point. Mancini analysis (D² vs C) must be read "
             "after the plateau; reading earlier needs the Fahey–McKelvey "
             "(D vs log C) method.")
    with tab3:
        c1, c2 = st.columns(2)
        r_nm = c1.number_input("Hydrodynamic radius (nm)", 0.5, 50.0, 5.3)
        T = c1.number_input("Temperature (°C)", 4.0, 45.0, 25.0)
        eta = c1.number_input("Viscosity (mPa·s)", 0.3, 3.0, 0.89)
        D = sc.stokes_einstein(r_nm * 1e-9, T + 273.15, eta * 1e-3)
        c2.metric("D (free solution)", f"{D * 1e4:.2e} cm²/s")
        st.latex(r"D=\frac{k_BT}{6\pi\eta r_h}")
        st.caption("Larger proteins and higher viscosity diffuse more "
                   "slowly, so rings and lines form later and closer to "
                   "the source.")


# ======================================================================
# LEARN
# ======================================================================
elif page == "Learn":
    head("Learn", "Concepts behind SRID, DID and diffusion.")
    t1, t2, t3, t4 = st.tabs(["SRID", "DID", "Fick's laws", "Glossary"])
    with t1:
        st.markdown(THEORY["SRID"])
        demo_c = np.array([0.5, 1, 1.5, 2, 2.5])
        d = np.sqrt(1600 + 5100 * demo_c)
        ex = pd.DataFrame({"conc": demo_c, "D": d, "D²": d ** 2})
        c1, c2 = st.columns(2)
        c1.markdown("**Diameter is curved…**")
        c1.altair_chart(alt.Chart(ex).mark_line(point=True, color=AMBER)
                        .encode(x="conc", y="D").properties(height=220))
        c2.markdown("**…but D² is linear.**")
        c2.altair_chart(alt.Chart(ex).mark_line(point=True, color=TEAL)
                        .encode(x="conc", y="D²").properties(height=220))
    with t2:
        st.markdown(THEORY["DID"])
        cols = st.columns(3)
        for col, kind in zip(cols, ["identity", "partial_identity",
                                    "non_identity"]):
            col.pyplot(did_schematic(kind))
    with t3:
        st.markdown(THEORY["FICK"])
    with t4:
        q = st.text_input("Search terms")
        for term, text in GLOSSARY.items():
            if q.lower() in term.lower() or q.lower() in text.lower():
                st.markdown(f"**{term}** – {text}")


# ======================================================================
# QUIZ
# ======================================================================
elif page == "Quiz":
    head("Quiz", f"{len(QUIZ)} questions across SRID, DID and diffusion.")
    topics = sorted({q["topic"] for q in QUIZ})
    chosen = st.multiselect("Topics", topics, default=topics)
    items = [(i, q) for i, q in enumerate(QUIZ) if q["topic"] in chosen]

    if not items:
        st.info("Select at least one topic.")
    elif not st.session_state.quiz_submitted:
        nonce = st.session_state.quiz_nonce
        with st.form(f"quiz_form_{nonce}"):
            for n, (i, q) in enumerate(items, 1):
                st.markdown(f"**{n}. {q['q']}**  \n:gray[{q['topic']}]")
                st.radio("Answer", q["options"], index=None,
                         key=f"q{nonce}_{i}", label_visibility="collapsed")
                st.write("")
            sent = st.form_submit_button("Submit answers", type="primary")
        if sent:
            st.session_state.quiz_answers = {
                i: st.session_state.get(f"q{nonce}_{i}") for i, _ in items}
            st.session_state.quiz_items = [i for i, _ in items]
            st.session_state.quiz_submitted = True
            st.rerun()
    else:
        ids = st.session_state.get("quiz_items", [i for i, _ in items])
        rows = []
        for i in ids:
            q = QUIZ[i]
            given = st.session_state.quiz_answers.get(i)
            right = q["options"][q["answer"]]
            rows.append({"idx": i, "topic": q["topic"], "given": given,
                         "right": right, "correct": given == right})
        res = pd.DataFrame(rows)
        score = int(res["correct"].sum())
        a, b, c = st.columns(3)
        a.metric("Score", f"{score} / {len(res)}")
        b.metric("Percentage", f"{100 * score / len(res):.0f}%")
        c.metric("Unanswered", int(res["given"].isna().sum()))

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("##### Overall")
            pie = pd.DataFrame({"result": ["Correct", "Incorrect"],
                                "n": [score, len(res) - score]})
            st.altair_chart(pie_chart(pie, "result", "n", [TEAL, RED]))
        with g2:
            st.markdown("##### By topic")
            by = res.groupby("topic")["correct"].mean().mul(100) \
                .reset_index()
            by.columns = ["topic", "percent"]
            st.altair_chart(bar_chart(by, "topic", "percent", "% correct"))

        st.markdown("#### Review")
        for _, r in res.iterrows():
            q = QUIZ[int(r["idx"])]
            mark = "Correct" if r["correct"] else "Incorrect"
            with st.expander(f"{mark} – {q['q']}"):
                st.write(f"Your answer: {r['given'] or 'no answer'}")
                st.write(f"Correct answer: **{r['right']}**")
                st.write(q["why"])
        if st.button("Try again"):
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = {}
            st.session_state.quiz_nonce += 1
            st.rerun()


# ======================================================================
# REPORTS
# ======================================================================
elif page == "Reports & Export":
    head("Reports & Export", "Download results and review the methods "
         "and limitations.")
    res = st.session_state.srid_results
    cal = st.session_state.calibration
    S = st.session_state.srid
    D = st.session_state.did

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### SRID")
        if res is None:
            st.write("No SRID results yet.")
        else:
            st.dataframe(res, hide_index=True)
            st.download_button("Results (CSV)", res.to_csv(index=False)
                               .encode(), "srid_results.csv", "text/csv")
            payload = {
                "calibration": (cal.__dict__ | {"equation": cal.equation}
                                if cal else None),
                "plate": {k: v for k, v in S["report"].items()
                          if k != "measurements"},
                "results": json.loads(res.to_json(orient="records")),
            }
            st.download_button("Full report (JSON)",
                               json.dumps(payload, indent=2, default=str),
                               "srid_report.json", "application/json")
    with c2:
        st.markdown("#### DID")
        if D is None:
            st.write("No DID plate analysed.")
        else:
            rep = D["report"]
            st.write(f"Wells: {rep['well_candidates']} · line candidates: "
                     f"{rep['line_candidates']}")
            st.write("Interpretation: "
                     f"**{D['interp'].replace('_', ' ')}**")
            st.download_button("DID report (JSON)",
                               json.dumps({**{k: v for k, v in rep.items()
                                              if k != "line_candidates_data"},
                                           "researcher_notes": D["comment"]},
                                          indent=2, default=str),
                               "did_report.json", "application/json")

    st.markdown("#### Methods")
    st.markdown("""
- **Detection:** CLAHE contrast enhancement, Hough circle and contour
  detection for wells; circular edge support (Canny) for ring candidates.
- **Calibration:** least-squares line of D² vs C (Mancini) or D vs log₁₀C
  (Fahey–McKelvey). At least three standards required.
- **Concentration:** inverse prediction from the line, multiplied by the
  dilution factor, flagged *extrapolated* outside the standards' range.
- **Provenance:** every diameter is labelled *researcher* or
  *auto-candidate*.
""")
    st.markdown("#### Limitations")
    note("Image detection can pick up bubbles, scratches and labels. Pixel "
         "values need a scale (known well diameter) before conversion to "
         "mm. DID interpretation is always the researcher's. Synthetic "
         "teaching plates must never be mixed with experimental data.")

st.divider()
st.caption("IMMUNOVISION_X · Research and teaching software · Verify all "
           "image-derived measurements before scientific use.")
