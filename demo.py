"""Synthetic demo plates so students can explore the app without a real gel."""
import cv2
import numpy as np

# Wells 1-5 are standards; well 6 is an 'unknown' whose true value is 1.75.
DEMO_CONCENTRATIONS = [0.5, 1.0, 1.5, 2.0, 2.5, 1.75]   # mg/mL (synthetic)


def make_srid_demo(path, seed=7):
    """Six wells; ring D² is proportional to concentration (Mancini)."""
    rng = np.random.default_rng(seed)
    h, w = 620, 900
    yy, xx = np.mgrid[0:h, 0:w]
    base = 150 + 25 * np.exp(-(((xx - w / 2) / 500) ** 2 +
                               ((yy - h / 2) / 400) ** 2))
    img = base + rng.normal(0, 3, (h, w))
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.2)
    img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    well_r = 20
    k = 5100.0                        # D² = d_well² + k·C  (Mancini)
    centres = [(190 + 260 * i, 190 + 240 * j)
               for j in range(2) for i in range(3)]
    for (cx, cy), c in zip(centres, DEMO_CONCENTRATIONS):
        ring_r = int(round(0.5 * np.sqrt((2 * well_r) ** 2 + k * c)))
        cv2.circle(img, (cx, cy), ring_r, (205, 205, 205), 3)
        cv2.circle(img, (cx, cy), ring_r - 3, (185, 185, 185), 1)
        cv2.circle(img, (cx, cy), well_r, (70, 70, 70), -1)
        cv2.circle(img, (cx, cy), well_r, (30, 30, 30), 2)
    cv2.imwrite(str(path), img)
    return path


def make_did_demo(path, seed=3):
    """Central antibody well + two antigen wells with a fused (identity) arc."""
    rng = np.random.default_rng(seed)
    h, w = 620, 760
    img = 160 + rng.normal(0, 3, (h, w))
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.2)
    img = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8),
                       cv2.COLOR_GRAY2BGR)
    wells = {"Ab": (380, 310), "Ag1": (200, 140), "Ag2": (560, 140),
             "Ag3": (380, 520)}
    for name, (x, y) in wells.items():
        cv2.circle(img, (x, y), 28, (75, 75, 75), -1)
        cv2.circle(img, (x, y), 28, (30, 30, 30), 2)
    # Fused arc between Ab and Ag1/Ag2 (identity)
    pts = np.array([[200, 215], [290, 195], [380, 185], [470, 195],
                    [560, 215]], np.int32)
    cv2.polylines(img, [pts], False, (215, 215, 215), 3)
    cv2.line(img, (380, 345), (380, 485), (215, 215, 215), 3)
    cv2.imwrite(str(path), img)
    return path
