
import streamlit as st
import pandas as pd
import numpy as np
import json
import tempfile
import inspect
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from analysis_engine import analyse_srid, analyse_did

st.set_page_config(
    page_title="IMMUNOVISION_X",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #f5f7fa; }
    section[data-testid="stSidebar"] { background-color: #eaf0f5; }
    .main-title {
        font-size: 2.15rem;
        font-weight: 700;
        color: #17324d;
        letter-spacing: -0.5px;
    }
    .sub-title { color: #60758a; font-size: 1rem; }
    .section-title {
        color: #17324d;
        font-weight: 650;
        font-size: 1.25rem;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #dce4ec;
        padding: 14px;
        border-radius: 9px;
    }
    .notice {
        background: #edf3f8;
        border-left: 4px solid #527895;
        padding: 12px 15px;
        border-radius: 5px;
        color: #263f55;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">IMMUNOVISION_X</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Integrated computational and experimental analysis '
    'of antigen–antibody precipitation</div>',
    unsafe_allow_html=True
)
st.divider()

with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Select workspace",
        ["Overview", "SRID Analysis", "DID Analysis",
         "Calibration", "Reports & Methods"]
    )
    st.divider()
    st.caption("IMMUNOVISION_X | Research workspace")
    st.caption("Experimental results require researcher verification.")

# Persistent session data
if "srid_df" not in st.session_state:
    st.session_state.srid_df = None
if "srid_report" not in st.session_state:
    st.session_state.srid_report = None
if "srid_image" not in st.session_state:
    st.session_state.srid_image = None
if "did_result" not in st.session_state:
    st.session_state.did_result = None
if "calibration" not in st.session_state:
    st.session_state.calibration = None


def save_upload(uploaded_file, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / uploaded_file.name
    path.write_bytes(uploaded_file.getvalue())
    return path


def find_report(output_dir, assay):
    candidates = list(Path(output_dir).rglob(f"{assay}_report.json"))
    if not candidates:
        candidates = list(Path(output_dir).rglob("*report*.json"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def show_output_images(output_dir):
    images = list(Path(output_dir).rglob("*.png"))
    if images:
        st.markdown("#### Generated image outputs")
        for image_path in images[:6]:
            try:
                st.image(str(image_path), caption=image_path.name,
                         use_container_width=True)
            except Exception:
                pass


def make_measurement_table(report):
    rows = report.get("measurements", [])
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Preserve the engine's candidate measurements.
    if "candidate_ring_diameter_px" not in df.columns:
        df["candidate_ring_diameter_px"] = np.nan

    if "verified_ring_diameter_px" not in df.columns:
        df["verified_ring_diameter_px"] = np.nan

    if "researcher_verified" not in df.columns:
        df["researcher_verified"] = False

    df["final_diameter_px"] = df["verified_ring_diameter_px"].where(
        df["verified_ring_diameter_px"].notna(),
        df["candidate_ring_diameter_px"]
    )

    df["D2_px2"] = df["final_diameter_px"] ** 2
    return df


if page == "Overview":
    st.markdown("### Research dashboard")
    st.markdown(
        "A computational workspace for reviewing precipitin patterns "
        "obtained through double immunodiffusion and single radial "
        "immunodiffusion."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Assay modules", "02")
    c2.metric("Analysis approach", "Image-based")
    c3.metric("Verification", "Researcher-led")

    st.markdown("### Project modules")
    a, b = st.columns(2)
    with a:
        st.markdown("#### Single Radial Immunodiffusion")
        st.write(
            "Identify well locations and candidate precipitin rings. "
            "Review ring diameters and calculate squared diameters (D²)."
        )
    with b:
        st.markdown("#### Double Immunodiffusion")
        st.write(
            "Review antigen–antibody precipitation patterns and "
            "record researcher interpretation of precipitin lines."
        )

    st.info(
        "Use experimental images for analysis. Candidate detections are "
        "not confirmed measurements until manually reviewed."
    )

elif page == "SRID Analysis":
    st.markdown("### Single Radial Immunodiffusion (SRID)")
    st.write(
        "Upload an experimental plate image to run the existing "
        "computational detection engine."
    )

    uploaded = st.file_uploader(
        "Upload SRID experimental image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="srid_upload"
    )

    if uploaded:
        st.image(uploaded, caption="Uploaded SRID image",
                 use_container_width=True)

        if st.button("Analyse SRID image", type="primary"):
            with st.spinner("Processing image with the analysis engine..."):
                run_dir = Path(tempfile.mkdtemp(prefix="immunovision_srid_"))
                image_path = save_upload(uploaded, run_dir / "input")
                output_dir = run_dir / "output"
                output_dir.mkdir(parents=True, exist_ok=True)

                try:
                    result = analyse_srid(str(image_path), str(output_dir))
                    report_path = find_report(output_dir, "srid")

                    report = None
                    if report_path:
                        report = json.loads(report_path.read_text())

                    # Some engine versions may return a report directly.
                    if report is None and isinstance(result, dict):
                        report = result.get("report", result)

                    if isinstance(report, dict) and "measurements" in report:
                        st.session_state.srid_report = report
                        st.session_state.srid_df = make_measurement_table(report)
                        st.session_state.srid_image = str(image_path)
                        st.success("SRID analysis completed. Review the candidate measurements below.")
                    else:
                        st.warning(
                            "The engine ran, but a standard SRID report "
                            "could not be located. Check the generated outputs."
                        )
                        st.write("Engine return:", result)
                        show_output_images(output_dir)

                except Exception as exc:
                    st.error(f"SRID analysis error: {exc}")
                    st.exception(exc)

    if st.session_state.srid_report is not None:
        report = st.session_state.srid_report
        df = st.session_state.srid_df.copy()

        st.divider()
        st.markdown("#### Analysis summary")
        x, y, z = st.columns(3)
        x.metric("Well candidates", report.get("total_well_candidates", len(df)))
        y.metric("Ring candidates", report.get("ring_candidates", "Not specified"))
        z.metric(
            "Calibration",
            "Available" if report.get("calibration_available") else "Not available"
        )

        if report.get("notice"):
            st.info(str(report["notice"]))

        st.markdown("#### Measurement review")
        st.caption(
            "Enter a verified diameter only after inspecting the actual "
            "precipitin ring. Do not use the well diameter as the ring diameter."
        )

        editable_columns = [
            col for col in [
                "well_number", "panel", "candidate_ring_diameter_px",
                "verified_ring_diameter_px", "researcher_verified"
            ] if col in df.columns
        ]

        edited = st.data_editor(
            df[editable_columns],
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="srid_editor"
        )

        # Recalculate from the researcher-verified diameter where supplied.
        final_diameter = edited["verified_ring_diameter_px"].where(
            edited["verified_ring_diameter_px"].notna(),
            edited["candidate_ring_diameter_px"]
        )
        result_df = edited.copy()
        result_df["final_diameter_px"] = final_diameter
        result_df["D2_px2"] = final_diameter ** 2

        st.markdown("#### Calculated measurements")
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        st.download_button(
            "Download SRID measurements (CSV)",
            result_df.to_csv(index=False).encode("utf-8"),
            file_name="IMMUNOVISION_X_SRID_measurements.csv",
            mime="text/csv"
        )

        st.caption(
            "D² is calculated as diameter × diameter. Values remain in px² "
            "unless image scale calibration is available."
        )

elif page == "DID Analysis":
    st.markdown("### Double Immunodiffusion (DID)")
    st.write(
        "Upload a DID plate image. The output should be reviewed by a "
        "researcher before assigning an antigen–antibody identity pattern."
    )

    uploaded = st.file_uploader(
        "Upload DID experimental image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="did_upload"
    )

    interpretation = st.text_area(
        "Researcher interpretation (optional)",
        placeholder="Record observed precipitin-line pattern and interpretation..."
    )

    if uploaded:
        st.image(uploaded, caption="Uploaded DID image",
                 use_container_width=True)

        if st.button("Analyse DID image", type="primary"):
            with st.spinner("Processing DID image..."):
                run_dir = Path(tempfile.mkdtemp(prefix="immunovision_did_"))
                image_path = save_upload(uploaded, run_dir / "input")
                output_dir = run_dir / "output"
                output_dir.mkdir(parents=True, exist_ok=True)

                try:
                    result = analyse_did(
                        str(image_path), str(output_dir), interpretation
                    )
                    st.session_state.did_result = {
                        "result": result,
                        "output_dir": str(output_dir),
                        "interpretation": interpretation
                    }
                    st.success("DID image processing completed.")
                except Exception as exc:
                    st.error(f"DID analysis error: {exc}")
                    st.exception(exc)

    if st.session_state.did_result:
        did = st.session_state.did_result
        st.markdown("#### Analysis output")
        st.write("Engine result:")
        st.json(did["result"], expanded=False) if isinstance(
            did["result"], (dict, list)
        ) else st.write(did["result"])
        st.markdown("#### Researcher interpretation")
        st.write(did["interpretation"] or "No interpretation entered.")
        show_output_images(did["output_dir"])

elif page == "Calibration":
    st.markdown("### SRID calibration")
    st.write(
        "Enter measurements from known antigen standards. Concentration "
        "estimation is enabled only when valid standards are supplied."
    )

    st.info(
        "Use the same assay conditions and measurement units for standards "
        "and unknown samples. Do not create a calibration curve from "
        "unverified candidate rings."
    )

    standards = st.data_editor(
        pd.DataFrame({
            "standard_concentration": [None, None, None],
            "verified_ring_diameter": [None, None, None]
        }),
        num_rows="dynamic",
        use_container_width=True,
        key="calibration_editor"
    )

    if st.button("Fit calibration curve"):
        valid = standards.dropna()
        valid = valid[
            (valid["standard_concentration"] > 0) &
            (valid["verified_ring_diameter"] > 0)
        ].copy()

        if len(valid) < 2:
            st.error("Enter at least two valid standard measurements.")
        else:
            valid["D2"] = valid["verified_ring_diameter"] ** 2
            slope, intercept = np.polyfit(
                valid["standard_concentration"], valid["D2"], 1
            )
            predicted = slope * valid["standard_concentration"] + intercept
            ss_res = np.sum((valid["D2"] - predicted) ** 2)
            ss_tot = np.sum((valid["D2"] - valid["D2"].mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

            st.session_state.calibration = {
                "slope": float(slope),
                "intercept": float(intercept),
                "r2": float(r2),
                "standards": valid
            }

    if st.session_state.calibration:
        cal = st.session_state.calibration
        st.markdown("#### Fitted calibration")
        st.metric("R²", f"{cal['r2']:.4f}")
        st.write(f"D² = {cal['slope']:.6g} × concentration + {cal['intercept']:.6g}")

        fig, ax = plt.subplots()
        standards_df = cal["standards"]
        ax.scatter(
            standards_df["standard_concentration"],
            standards_df["D2"], label="Standards"
        )
        x_values = np.linspace(
            standards_df["standard_concentration"].min(),
            standards_df["standard_concentration"].max(), 100
        )
        ax.plot(
            x_values, cal["slope"] * x_values + cal["intercept"],
            label="Linear fit"
        )
        ax.set_xlabel("Standard antigen concentration")
        ax.set_ylabel("Squared ring diameter (D²)")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)

        st.caption(
            "This is a fitted relationship from the standards entered here. "
            "Check the assay's suitability and linear range before using it "
            "to estimate unknown concentrations."
        )

elif page == "Reports & Methods":
    st.markdown("### Reports and scientific methods")

    st.markdown("#### Measurement principles")
    st.write(
        "**SRID:** The precipitin ring diameter is measured, then squared "
        "to obtain D². Concentration is estimated only from a valid "
        "standard calibration curve."
    )
    st.write(
        "**DID:** Precipitin-line patterns are reviewed to assess possible "
        "identity, partial identity, or non-identity. Final interpretation "
        "requires visual confirmation."
    )

    st.markdown("#### Data integrity")
    st.write(
        "- Automatic detections are candidates, not confirmed results."
        "\n- Missing ring measurements must remain missing until reviewed."
        "\n- Pixel measurements require scale calibration before conversion "
        "to physical units."
        "\n- Experimental and simulated data must be labelled separately."
    )

    st.markdown("#### Project title")
    st.write(
        "Integrated Experimental and Computational Analysis of "
        "Antigen–Antibody Precipitation: A Comparative Study of "
        "Double Immunodiffusion and Single Radial Immunodiffusion."
    )

st.divider()
st.caption(
    "IMMUNOVISION_X | Research support software | "
    "Verify all image-derived measurements before scientific use."
)
