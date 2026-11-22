
import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path


def load_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("The uploaded file could not be decoded as an image.")
    return image


def prepare_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    ).apply(gray)
    return gray


def split_panels(image):
    """
    For a three-panel A/B/C figure, split into three vertical panels
    when the image is substantially wider than it is tall.
    Otherwise, analyse the image as one panel.
    """
    h, w = image.shape[:2]

    if w / max(h, 1) > 1.45:
        cuts = [0, w // 3, (2 * w) // 3, w]
        panels = []

        for i in range(3):
            x1, x2 = cuts[i], cuts[i + 1]
            margin = int((x2 - x1) * 0.035)
            panels.append(
                (image[:, x1 + margin:x2 - margin], x1 + margin, 0)
            )

        return panels

    return [(image, 0, 0)]


def detect_wells(panel, min_radius_px=12):
    """
    Detect well candidates while rejecting tiny text/label artefacts.
    The default 12-pixel threshold is targeted to the current dataset.
    Adjust it for images with smaller genuine wells.
    """
    gray = prepare_image(panel)
    h, w = gray.shape

    min_dim = min(h, w)
    min_radius = max(int(min_radius_px), 5)
    max_radius = max(min_radius + 2, int(min_dim * 0.075))

    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    candidates = []

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=max(20, min_radius * 2),
        param1=100,
        param2=28,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    if circles is not None:
        for x, y, r in np.round(circles[0]).astype(int):
            if r >= min_radius:
                candidates.append((int(x), int(y), int(r), 0.70))

    edges = cv2.Canny(gray, 40, 120)
    contours, _ = cv2.findContours(
        edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

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
            candidates.append(
                (int(round(x)), int(round(y)), radius,
                 float(circularity))
            )

    # Merge duplicate Hough and contour detections.
    candidates.sort(key=lambda item: item[3], reverse=True)
    accepted = []

    for candidate in candidates:
        x, y, r, score = candidate

        duplicate = any(
            np.hypot(x - ax, y - ay) <
            max(10, min(r, ar) * 0.85)
            for ax, ay, ar, _ in accepted
        )

        if not duplicate:
            accepted.append(candidate)

    # Exclude detections close to panel borders.
    accepted = [
        c for c in accepted
        if c[0] > max_radius
        and c[1] > max_radius
        and c[0] < w - max_radius
        and c[1] < h - max_radius
    ]

    accepted.sort(key=lambda c: (c[1], c[0]))
    return accepted



def ring_support(edges, x, y, radius):
    """
    Calculate circular edge support from a precomputed Canny edge map.
    This produces a candidate score, not biological confirmation.
    """
    h, w = edges.shape
    angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
    supported = 0

    for angle in angles:
        found = False

        for delta in (-2, -1, 0, 1, 2):
            r = radius + delta
            px = int(round(x + r * np.cos(angle)))
            py = int(round(y + r * np.sin(angle)))

            if 0 <= px < w and 0 <= py < h:
                if edges[py, px] > 0:
                    found = True
                    break

        if found:
            supported += 1

    return supported / len(angles)



def analyse_srid(image_path, output_dir):
    """
    SRID image analysis.
    Outputs candidate well and ring measurements in pixels.
    Candidate measurements require manual verification.
    """
    image = load_image(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated = image.copy()
    records = []
    panel_data = split_panels(image)
    global_well_number = 0

    for panel_number, (panel, offset_x, offset_y) in enumerate(
        panel_data, start=1
    ):
        gray = prepare_image(panel)

        # Calculate edges once per panel, not once per radius.
        blurred = cv2.GaussianBlur(gray, (5, 5), 1)
        edges = cv2.Canny(blurred, 35, 110)

        wells = detect_wells(panel, min_radius_px=12)

        for x, y, well_radius, well_score in wells:
            global_well_number += 1

            min_ring = max(
                well_radius + 5,
                int(round(well_radius * 1.35))
            )
            max_ring = int(round(well_radius * 3.7))

            candidates = []

            for radius in range(min_ring, max_ring + 1):
                support = ring_support(edges, x, y, radius)
                candidates.append((support, radius))

            candidates.sort(key=lambda item: item[0], reverse=True)

            best_support, best_radius = (
                candidates[0] if candidates else (0.0, None)
            )

            # Conservative candidate threshold.
            # This does not establish that the edge is a true precipitin ring.
            ring_detected = (
                best_radius is not None
                and best_support >= 0.28
            )

            gx = int(x + offset_x)
            gy = int(y + offset_y)

            cv2.circle(
                annotated,
                (gx, gy),
                int(well_radius),
                (255, 80, 180),
                2
            )

            if ring_detected:
                cv2.circle(
                    annotated,
                    (gx, gy),
                    int(best_radius),
                    (0, 165, 255),
                    2
                )

            candidate_diameter = (
                int(best_radius * 2) if ring_detected else None
            )

            # D squared is only a candidate image-derived value.
            candidate_d2 = (
                int(candidate_diameter ** 2)
                if candidate_diameter is not None else None
            )

            records.append({
                "well_number": int(global_well_number),
                "panel": int(panel_number),
                "center_x_px": gx,
                "center_y_px": gy,
                "well_radius_px": int(well_radius),
                "well_diameter_px": int(well_radius * 2),
                "candidate_ring_radius_px": (
                    int(best_radius) if ring_detected else None
                ),
                "candidate_ring_diameter_px": candidate_diameter,
                "candidate_ring_d2_px2": candidate_d2,
                "angular_support": round(float(best_support), 3),
                "ring_candidate_detected": bool(ring_detected),
                "review_required": True,
                "measurement_status": (
                    "CANDIDATE_REQUIRES_MANUAL_VERIFICATION"
                    if ring_detected else "NO_CONFIRMED_RING"
                )
            })

    csv_path = output_dir / "srid_measurements.csv"
    image_path_out = output_dir / "srid_annotated.png"
    json_path = output_dir / "srid_report.json"

    pd.DataFrame(records).to_csv(csv_path, index=False)
    cv2.imwrite(str(image_path_out), annotated)

    ring_count = sum(
        1 for r in records if r["ring_candidate_detected"]
    )

    report = {
        "assay": "SRID",
        "total_well_candidates": len(records),
        "ring_candidates": ring_count,
        "measurement_unit": "pixels",
        "d2_unit": "pixels_squared",
        "calibration_available": False,
        "concentration_calculation_available": False,
        "manual_verification_required": True,
        "notice": (
            "Ring measurements and D-squared values are image-derived "
            "candidates, not validated experimental measurements. "
            "Verify ring boundaries manually. Antigen concentration "
            "requires experimental calibration standards."
        ),
        "measurements": records
    }

    json_path.write_text(
        json.dumps(report, indent=2, default=lambda value: (
            value.item() if hasattr(value, "item")
            else value.tolist() if hasattr(value, "tolist")
            else str(value)
        )),
        encoding="utf-8"
    )

    return report



def analyse_did(image_path, output_dir, interpretation="pending"):
    image = load_image(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated = image.copy()
    gray = prepare_image(image)

    wells = detect_wells(image)

    edges = cv2.Canny(
        cv2.GaussianBlur(gray, (5, 5), 1),
        35,
        110
    )

    # Join nearby broken portions of possible precipitin lines.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    joined = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    contours, _ = cv2.findContours(
        joined,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    line_candidates = []

    for contour in contours:
        length = cv2.arcLength(contour, False)

        if length < 35:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Avoid tiny text fragments and very large image borders.
        if w < 8 and h < 8:
            continue

        if w > image.shape[1] * 0.95:
            continue

        if h > image.shape[0] * 0.95:
            continue

        line_candidates.append({
            "x": int(x),
            "y": int(y),
            "width": int(w),
            "height": int(h),
            "contour_length_px": round(float(length), 2)
        })

        cv2.drawContours(
            annotated,
            [contour],
            -1,
            (60, 220, 80),
            2
        )

    for number, (x, y, radius, score) in enumerate(wells, start=1):
        cv2.circle(
            annotated,
            (x, y),
            radius,
            (255, 80, 180),
            2
        )

        cv2.putText(
            annotated,
            f"W{number}",
            (x + radius, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 80, 180),
            1
        )

    image_path_out = output_dir / "did_annotated.png"
    csv_path = output_dir / "did_line_candidates.csv"
    json_path = output_dir / "did_report.json"

    cv2.imwrite(str(image_path_out), annotated)
    pd.DataFrame(line_candidates).to_csv(csv_path, index=False)

    allowed = {
        "pending",
        "identity",
        "partial_identity",
        "non_identity"
    }

    if interpretation not in allowed:
        interpretation = "pending"

    report = {
        "assay": "DID",
        "well_candidates": len(wells),
        "line_candidates": len(line_candidates),
        "pattern_interpretation": interpretation,
        "interpretation_status": (
            "researcher_review_required"
            if interpretation == "pending"
            else "researcher_entered"
        ),
        "possible_patterns": {
            "identity": "Fused continuous precipitin lines.",
            "partial_identity": "Fused lines with a spur.",
            "non_identity": "Crossing precipitin lines."
        },
        "notice": (
            "Image processing identifies candidate wells and edges. "
            "It does not independently confirm antigen identity, "
            "partial identity, or non-identity. Confirm the pattern "
            "by visual inspection of the experimental plate."
        ),
        "line_candidates_data": line_candidates
    }

    json_path.write_text(json.dumps(
            report,
            indent=2,
            default=lambda value: (
                value.item() if hasattr(value, "item")
                else value.tolist() if hasattr(value, "tolist")
                else str(value)
            )
        ))

    return report
