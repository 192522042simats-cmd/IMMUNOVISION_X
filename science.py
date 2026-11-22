"""
Quantitative immunodiffusion science for IMMUNOVISION_X.

Pure NumPy/pandas: no Streamlit, so it can be unit-tested.
"""
from dataclasses import dataclass
from math import erfc, pi, sqrt

import numpy as np
import pandas as pd

K_BOLTZMANN = 1.380649e-23  # J/K


# ----------------------------------------------------------------------
# Scale (pixels -> mm)
# ----------------------------------------------------------------------
def px_per_mm_from_reference(reference_px, reference_mm):
    if reference_px is None or reference_mm is None:
        return None
    if reference_px <= 0 or reference_mm <= 0:
        return None
    return reference_px / reference_mm


# ----------------------------------------------------------------------
# SRID calibration
# ----------------------------------------------------------------------
METHODS = {
    "mancini": "Mancini end-point (D² vs concentration)",
    "fahey": "Fahey–McKelvey kinetic (D vs log10 concentration)",
}


def _transform(diameter, method, well_diameter=0.0):
    d = np.asarray(diameter, dtype=float)
    if method == "mancini":
        return d ** 2 - well_diameter ** 2   # well-corrected D²
    return d


def _x_transform(conc, method):
    c = np.asarray(conc, dtype=float)
    return c if method == "mancini" else np.log10(c)


@dataclass
class Calibration:
    method: str
    slope: float
    intercept: float
    r2: float
    n: int
    well_diameter: float
    conc_min: float
    conc_max: float
    y_min: float
    y_max: float
    unit_diameter: str = "px"
    unit_conc: str = "mg/mL"

    @property
    def equation(self):
        if self.method == "mancini":
            return (f"D² = {self.slope:.5g} × C + {self.intercept:.5g}")
        return f"D = {self.slope:.5g} × log10(C) + {self.intercept:.5g}"

    def quality(self):
        if self.r2 >= 0.99:
            return "Excellent"
        if self.r2 >= 0.97:
            return "Acceptable"
        return "Poor – repeat or review standards"


def fit_calibration(conc, diameter, method="mancini", well_diameter=0.0,
                    unit_diameter="px", unit_conc="mg/mL"):
    conc = np.asarray(conc, float)
    diameter = np.asarray(diameter, float)
    ok = np.isfinite(conc) & np.isfinite(diameter) & (conc > 0) \
        & (diameter > 0)
    conc, diameter = conc[ok], diameter[ok]
    if len(conc) < 3:
        raise ValueError("At least 3 valid standards are needed for a "
                         "reliable calibration line.")
    if len(np.unique(conc)) < 2:
        raise ValueError("Standards must have at least two different "
                         "concentrations.")
    x = _x_transform(conc, method)
    y = _transform(diameter, method, well_diameter)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    return Calibration(method, float(slope), float(intercept), float(r2),
                       int(len(conc)), float(well_diameter),
                       float(conc.min()), float(conc.max()),
                       float(y.min()), float(y.max()),
                       unit_diameter, unit_conc)


def predict_concentration(cal, diameter, dilution_factor=1.0):
    """
    Returns (concentration, flag) arrays. flag is 'ok', 'extrapolated'
    or 'invalid'. Extrapolated values lie outside the standards' range
    and should be re-run at another dilution.
    """
    d = np.atleast_1d(np.asarray(diameter, float))
    y = _transform(d, cal.method, cal.well_diameter)
    if cal.slope == 0:
        raise ValueError("Calibration slope is zero.")
    x = (y - cal.intercept) / cal.slope
    conc = x if cal.method == "mancini" else 10 ** x
    flags = []
    out = []
    for c, yy, dd in zip(conc, y, d):
        if not np.isfinite(dd) or not np.isfinite(c) or c <= 0:
            out.append(np.nan)
            flags.append("invalid")
            continue
        c = c * dilution_factor
        out.append(c)
        in_range = cal.y_min * 0.98 <= yy <= cal.y_max * 1.02
        flags.append("ok" if in_range else "extrapolated")
    return np.array(out), flags


def calibration_curve_points(cal, n=100):
    cs = np.linspace(cal.conc_min * 0.8, cal.conc_max * 1.2, n)
    x = _x_transform(cs, cal.method)
    return cs, cal.slope * x + cal.intercept


# ----------------------------------------------------------------------
# Fick's laws & diffusion
# ----------------------------------------------------------------------
def fick_first_flux(D, dC_dx):
    """J = -D dC/dx   (D in m²/s, dC/dx in mol/m⁴ -> J in mol/m²/s)."""
    return -D * dC_dx


def diffusion_length(D, t):
    """Characteristic diffusion distance x ≈ 2√(D t)."""
    return 2.0 * sqrt(D * t)


def fick_second_profile(x, t, D, C0):
    """
    1-D semi-infinite solution of Fick's second law with constant source:
        C(x,t) = C0 · erfc( x / (2√(D t)) )
    """
    x = np.asarray(x, float)
    if t <= 0:
        return np.where(x <= 0, C0, 0.0)
    z = x / (2.0 * sqrt(D * t))
    return C0 * np.array([erfc(v) for v in np.atleast_1d(z)])


def radial_profile(r, t, D, M, h=1.0):
    """
    Radial diffusion from an instantaneous point source in a gel of
    thickness h (2-D Gaussian):  C(r,t) = M / (4π D t h) · exp(-r²/4Dt)
    """
    r = np.asarray(r, float)
    return M / (4 * pi * D * t * h) * np.exp(-r ** 2 / (4 * D * t))


def stokes_einstein(radius_m, temp_K=298.15, viscosity_Pa_s=0.00089):
    """D = kT / (6 π η r_h)"""
    return K_BOLTZMANN * temp_K / (6 * pi * viscosity_Pa_s * radius_m)


# Typical free-solution diffusion coefficients, cm²/s (room temp, aqueous).
DIFFUSION_REFERENCE = {
    "IgG (~150 kDa)": 4.0e-7,
    "IgM (~900 kDa)": 2.6e-7,
    "Albumin (~66 kDa)": 6.1e-7,
    "Transferrin (~80 kDa)": 5.0e-7,
}


# ----------------------------------------------------------------------
# DID helper
# ----------------------------------------------------------------------
def equivalence_note(line_position_fraction):
    """
    line_position_fraction: 0 = at antigen well, 1 = at antibody well.
    Precipitin line forms nearer the well holding the *less concentrated*
    / slower-diffusing reagent.
    """
    f = float(line_position_fraction)
    if f < 0.4:
        return ("Line close to the antigen well: antigen is limiting "
                "(antibody in excess).")
    if f > 0.6:
        return ("Line close to the antibody well: antibody is limiting "
                "(antigen in excess).")
    return "Line near the midpoint: approximate equivalence zone."


def summarise_srid(df, px_per_mm=None):
    """Add mm columns when a scale is known."""
    out = df.copy()
    if px_per_mm:
        out["ring_diameter_mm"] = out["final_diameter_px"] / px_per_mm
        out["D2_mm2"] = out["ring_diameter_mm"] ** 2
    return out


# ----------------------------------------------------------------------
# End-to-end SRID results table
# ----------------------------------------------------------------------
def build_srid_results(table, well_diameter_px_median, known_well_mm=None,
                       method="mancini", subtract_well=False,
                       dilution=1.0, unit_conc="mg/mL"):
    """
    table columns: well_number, role ('Standard'|'Unknown'|'Ignore'),
    standard_conc, candidate_diameter_px, verified_diameter_px.

    Returns (results_df, calibration_or_None, notes_list).
    Verified values always override candidates; missing stays missing.
    """
    notes = []
    df = table.copy()
    for col in ("standard_conc", "candidate_diameter_px",
                "verified_diameter_px"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    final_px = df["verified_diameter_px"].where(
        df["verified_diameter_px"].notna(), df["candidate_diameter_px"])
    source = np.where(df["verified_diameter_px"].notna(), "researcher",
                      np.where(df["candidate_diameter_px"].notna(),
                               "auto-candidate", "missing"))

    px_per_mm = None
    if known_well_mm and known_well_mm > 0 and well_diameter_px_median > 0:
        px_per_mm = well_diameter_px_median / known_well_mm
    unit = "mm" if px_per_mm else "px"
    final = final_px / px_per_mm if px_per_mm else final_px
    well_d = (well_diameter_px_median / px_per_mm if px_per_mm
              else well_diameter_px_median) if subtract_well else 0.0

    out = pd.DataFrame({
        "well": df["well_number"],
        "role": df["role"],
        f"diameter_{unit}": final,
        f"D2_{unit}2": final ** 2,
        "diameter_source": source,
        "standard_conc": df["standard_conc"],
        "estimated_conc": np.nan,
        "flag": "",
    })

    stds = df["role"].eq("Standard") & final.notna() & \
        df["standard_conc"].gt(0)
    cal = None
    if stds.sum() >= 3:
        try:
            cal = fit_calibration(df.loc[stds, "standard_conc"],
                                  final[stds], method, well_d,
                                  unit, unit_conc)
        except ValueError as exc:
            notes.append(str(exc))
    else:
        notes.append(f"{int(stds.sum())} usable standard(s) – at least 3 "
                     "standards with concentration and ring diameter are "
                     "needed to calibrate.")

    if cal is not None:
        unk = df["role"].eq("Unknown") & final.notna()
        if unk.any():
            conc, flags = predict_concentration(cal, final[unk], dilution)
            out.loc[unk, "estimated_conc"] = conc
            out.loc[unk, "flag"] = flags
        if cal.r2 < 0.97:
            notes.append(f"Calibration R² = {cal.r2:.3f} (below 0.97): "
                         "review ring measurements before trusting "
                         "concentrations.")
        if (out["flag"] == "extrapolated").any():
            notes.append("Some unknowns fall outside the standards' range; "
                         "re-run them at a different dilution.")
    out.loc[out["role"].eq("Standard"), "flag"] = ""
    return out, cal, notes
