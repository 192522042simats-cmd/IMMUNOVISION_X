"""
IMMUNOVISION_X analysis engine.

Image-based candidate detection for SRID (single radial immunodiffusion)
and DID (double immunodiffusion / Ouchterlony) plates.

All detections are *candidates*. The researcher verifies them in the app.
Missing values are returned as NaN (never None) so tables stay numeric.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Basic helpers
# ----------------------------------------------------------------------
def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("The uploaded file could not be decoded as an image.")
    return image


def downscale_if_large(image, max_side=1800):
    """Large phone/scanner photos make Hough detection slow and noisy."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image, 1.0
    scale = max_side / longest
    small = cv2.resize(image, (int(w * scale), int(h * scale)),
                       interpolation=cv2.INTER_AREA)
    return small, scale


def prepare_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def auto_min_radius(shape, floor=5):
    """Scale-aware minimum well radius (about 1.2 % of the short side)."""
    return max(floor, int(round(min(shape[:2]) * 0.012)))


def split_panels(image, mode="single"):
    """
    mode="single": analyse the whole image (default, safe for plate photos).
    mode="three":  split a wide A/B/C figure into three vertical panels.
    Splitting is never automatic: a landscape photo of one plate would
    otherwise be cut through its wells.
    """
    h, w = image.shape[:2]
    if mode == "three":
        cuts = [0, w // 3, (2 * w) // 3, w]
        panels = []
        for i in range(3):
            x1, x2 = cuts[i], cuts[i + 1]
            margin = int((x2 - x1) * 0.035)
            panels.append((image[:, x1 + margin:x2 - margin], x1 + margin, 0))
        return panels
    return [(image, 0, 0)]


# ----------------------------------------------------------------------
# Well detection
# ----------------------------------------------------------------------
def detect_wells(panel, min_radius_px=None):
    """
    Detect circular well candidates.

    min_radius_px=None -> chosen automatically from the image size, so the
    same code works for small web figures and large scanner images.
    """
    gray = prepare_image(panel)
    h, w = gray.shape
    min_dim = min(h, w)

    if min_radius_px is None:
        min_radius_px = auto_min_radius(gray.shape)
    min_radius = max(int(min_radius_px), 4)
    max_radius = max(min_radius + 2, int(min_dim * 0.075))

    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    candidates = []

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=max(20, min_radius * 2),
        param1=100, param2=28,
        minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            if r >= min_radius:
                candidates.append((int(x), int(y), int(r), 0.70))

    edges = cv2.Canny(gray, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0 or area <= 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        if circularity < 0.62:
            continue
        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = int(round(radius))
        if min_radius <= radius <= max_radius:
            candidates.append((int(round(x)), int(round(y)), radius,
                               float(circularity)))

    # Merge duplicate Hough / contour detections.
    candidates.sort(key=lambda c: c[3], reverse=True)
    accepted = []
    for x, y, r, score in candidates:
        duplicate = any(
            np.hypot(x - ax, y - ay) < max(10, min(r, ar) * 0.85)
            for ax, ay, ar, _ in accepted
        )
        if not duplicate:
            accepted.append((x, y, r, score))

    # Drop detections touching the panel border.
    accepted = [c for c in accepted
                if max_radius < c[0] < w - max_radius
                and max_radius < c[1] < h - max_radius]

    accepted.sort(key=lambda c: (c[1], c[0]))
    return accepted


# ----------------------------------------------------------------------
# Ring detection (vectorised: ~50x faster than the per-pixel loops)
# ----------------------------------------------------------------------
_ANGLES = np.linspace(0, 2 * np.pi, 96, endpoint=False)
_COS, _SIN = np.cos(_ANGLES), np.sin(_ANGLES)


def ring_support(edges, x, y, radius):
    """Fraction of the circle (+/-2 px) that lies on a Canny edge."""
    h, w = edges.shape
    found = np.zeros(len(_ANGLES), dtype=bool)
    for delta in (-2, -1, 0, 1, 2):
        r = radius + delta
        px = np.rint(x + r * _COS).astype(int)
        py = np.rint(y + r * _SIN).astype(int)
        ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        hit = np.zeros(len(_ANGLES), dtype=bool)
        hit[ok] = edges[py[ok], px[ok]] > 0
        found |= hit
    return float(found.mean())


def radial_profile_ring(gray, x, y, well_radius, max_radius):
    """
    Second opinion on the ring radius from the mean radial intensity
    profile: precipitin rings show up as a sharp change in brightness.
    Returns (radius, strength) or (nan, 0).
    """
    h, w = gray.shape
    radii = np.arange(int(well_radius * 1.3), int(max_radius) + 1)
    if len(radii) < 6:
        return np.nan, 0.0
    means = []
    for r in radii:
        px = np.rint(x + r * _COS).astype(int)
        py = np.rint(y + r * _SIN).astype(int)
        ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        means.append(gray[py[ok], px[ok]].mean() if ok.sum() > 20 else np.nan)
    means = np.array(means, dtype=float)
    if np.isnan(means).all():
        return np.nan, 0.0
    means = np.nan_to_num(means, nan=np.nanmean(means))
    grad = np.abs(np.gradient(cv2.GaussianBlur(
        means.reshape(-1, 1), (1, 5), 0).ravel()))
    i = int(np.argmax(grad))
    strength = float(grad[i] / (grad.mean() + 1e-6))
    return float(radii[i]), strength


# ----------------------------------------------------------------------
# JSON helper
# ----------------------------------------------------------------------
def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _clean(obj):
    """Replace NaN with None for JSON output."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    return obj


# ----------------------------------------------------------------------
# SRID
# ----------------------------------------------------------------------
def analyse_srid(image_path, output_dir, min_radius_px=None,
                 support_threshold=0.28, panels="single"):
    """
    SRID candidate analysis. Measurements are in pixels of the (possibly
    downscaled) analysed image; `scale_to_original` is reported so values
    can be converted back.
    """
    original = load_image(image_path)
    image, scale = downscale_if_large(original)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated = image.copy()
    records = []
    global_well_number = 0

    for panel_number, (panel, off_x, off_y) in enumerate(
            split_panels(image, panels), start=1):
        gray = prepare_image(panel)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1), 35, 110)
        wells = detect_wells(panel, min_radius_px=min_radius_px)
        # Wells are punched with one tool: snap outlier radii to the median
        # so a ring is never mistaken for a larger well.
        if len(wells) >= 3:
            med = int(np.median([c[2] for c in wells]))
            wells = [(x, y, med if r > 1.35 * med else r, s)
                     for x, y, r, s in wells]
        max_ring_limit = int(min(gray.shape) * 0.45)

        for x, y, well_radius, _ in wells:
            global_well_number += 1
            min_ring = max(well_radius + 5, int(round(well_radius * 1.35)))
            max_ring = min(int(round(well_radius * 3.7)), max_ring_limit)

            best_support, best_radius = 0.0, None
            for radius in range(min_ring, max_ring + 1):
                s = ring_support(edges, x, y, radius)
                if s > best_support:
                    best_support, best_radius = s, radius

            ring_detected = best_radius is not None and \
                best_support >= support_threshold

            prof_r, prof_strength = radial_profile_ring(
                gray, x, y, well_radius, max_ring)

            gx, gy = int(x + off_x), int(y + off_y)
            cv2.circle(annotated, (gx, gy), int(well_radius),
                       (255, 80, 180), 2)
            cv2.putText(annotated, str(global_well_number),
                        (gx - 6, gy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)
            if ring_detected:
                cv2.circle(annotated, (gx, gy), int(best_radius),
                           (0, 165, 255), 2)

            diameter = float(best_radius * 2) if ring_detected else np.nan
            records.append({
                "well_number": global_well_number,
                "panel": panel_number,
                "center_x_px": gx,
                "center_y_px": gy,
                "well_radius_px": int(well_radius),
                "well_diameter_px": int(well_radius * 2),
                "candidate_ring_radius_px":
                    float(best_radius) if ring_detected else np.nan,
                "candidate_ring_diameter_px": diameter,
                "candidate_ring_d2_px2": diameter ** 2 if ring_detected
                    else np.nan,
                "profile_ring_diameter_px": prof_r * 2,
                "profile_strength": round(prof_strength, 2),
                "angular_support": round(float(best_support), 3),
                "ring_candidate_detected": bool(ring_detected),
                "review_required": True,
            })

    annotated_path = output_dir / "srid_annotated.png"
    cv2.imwrite(str(annotated_path), annotated)

    df = pd.DataFrame(records)
    df.to_csv(output_dir / "srid_measurements.csv", index=False)

    report = {
        "assay": "SRID",
        "total_well_candidates": len(records),
        "ring_candidates": int(sum(r["ring_candidate_detected"]
                                   for r in records)),
        "measurement_unit": "pixels",
        "scale_to_original": scale,
        "image_size_analysed": [int(image.shape[1]), int(image.shape[0])],
        "min_radius_used_px": (min_radius_px if min_radius_px is not None
                               else auto_min_radius(image.shape)),
        "annotated_image": str(annotated_path),
        "notice": ("Rings and D² are image-derived candidates. Verify ring "
                   "boundaries manually; concentration needs calibration "
                   "standards."),
        "measurements": records,
    }
    (output_dir / "srid_report.json").write_text(
        json.dumps(_clean(report), indent=2, default=_json_default),
        encoding="utf-8")
    return report


# ----------------------------------------------------------------------
# DID
# ----------------------------------------------------------------------
DID_PATTERNS = {
    "pending": "Not yet interpreted.",
    "identity": "Fused continuous precipitin lines (identical antigens).",
    "partial_identity": "Fused lines with a spur (shared + unique epitopes).",
    "non_identity": "Crossing precipitin lines (unrelated antigens).",
    "no_precipitin": "No precipitin line formed (no reaction / out of "
                     "equivalence zone).",
}


def analyse_did(image_path, output_dir, interpretation="pending",
                min_radius_px=None):
    original = load_image(image_path)
    image, scale = downscale_if_large(original)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated = image.copy()
    gray = prepare_image(image)
    wells = detect_wells(image, min_radius_px=min_radius_px)

    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1), 35, 110)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    joined = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(joined, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)

    well_mask = np.zeros(gray.shape, np.uint8)
    for x, y, r, _ in wells:
        cv2.circle(well_mask, (x, y), int(r * 1.25), 255, -1)

    lines = []
    for contour in contours:
        length = cv2.arcLength(contour, False)
        if length < 35:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < 8 and h < 8:
            continue
        if w > image.shape[1] * 0.95 or h > image.shape[0] * 0.95:
            continue
        # Skip contours that are just well outlines.
        m = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(m, [contour], -1, 255, 2)
        overlap = cv2.countNonZero(cv2.bitwise_and(m, well_mask))
        if overlap > 0.7 * max(cv2.countNonZero(m), 1):
            continue
        lines.append({"line_id": len(lines) + 1, "x": int(x), "y": int(y),
                      "width": int(w), "height": int(h),
                      "contour_length_px": round(float(length), 2)})
        cv2.drawContours(annotated, [contour], -1, (60, 220, 80), 2)

    well_rows = []
    for n, (x, y, r, _) in enumerate(wells, start=1):
        cv2.circle(annotated, (x, y), r, (255, 80, 180), 2)
        cv2.putText(annotated, f"W{n}", (x + r + 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 180), 1)
        well_rows.append({"well": f"W{n}", "x_px": x, "y_px": y,
                          "radius_px": r})

    # Centre-to-centre distances (useful for diffusion / Fick's-law work).
    distances = []
    for i in range(len(well_rows)):
        for j in range(i + 1, len(well_rows)):
            a, b = well_rows[i], well_rows[j]
            distances.append({
                "from": a["well"], "to": b["well"],
                "distance_px": round(float(np.hypot(
                    a["x_px"] - b["x_px"], a["y_px"] - b["y_px"])), 1)})

    annotated_path = output_dir / "did_annotated.png"
    cv2.imwrite(str(annotated_path), annotated)
    pd.DataFrame(lines).to_csv(output_dir / "did_line_candidates.csv",
                               index=False)

    if interpretation not in DID_PATTERNS:
        interpretation = "pending"

    report = {
        "assay": "DID",
        "well_candidates": len(wells),
        "line_candidates": len(lines),
        "pattern_interpretation": interpretation,
        "interpretation_status": ("researcher_review_required"
                                  if interpretation == "pending"
                                  else "researcher_entered"),
        "possible_patterns": DID_PATTERNS,
        "annotated_image": str(annotated_path),
        "scale_to_original": scale,
        "notice": ("Image processing finds candidate wells and edges only. "
                   "Identity / partial identity / non-identity must be "
                   "confirmed by visual inspection of the plate."),
        "wells": well_rows,
        "well_distances": distances,
        "line_candidates_data": lines,
    }
    (output_dir / "did_report.json").write_text(
        json.dumps(_clean(report), indent=2, default=_json_default),
        encoding="utf-8")
    return report
