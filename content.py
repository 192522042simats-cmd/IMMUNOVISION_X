"""Educational content: quizzes, glossary, theory text."""

QUIZ = [
    {
        "topic": "SRID",
        "q": "In SRID (Mancini method), which quantity is proportional to antigen concentration at end-point?",
        "options": ["Ring diameter D", "Squared ring diameter D²", "log of ring diameter", "Well depth"],
        "answer": 1,
        "why": "At end-point diffusion the ring area (∝ D²) is proportional to the amount of antigen, so D² vs concentration is linear.",
    },
    {
        "topic": "SRID",
        "q": "Where is the antibody in an SRID plate?",
        "options": ["In the well", "Uniformly mixed into the agarose gel", "Only on the plate surface", "In a separate trough"],
        "answer": 1,
        "why": "Antibody is incorporated into the gel; antigen diffuses radially out of the well and precipitates where it meets antibody at equivalence.",
    },
    {
        "topic": "SRID",
        "q": "Why must unknown samples fall inside the range of the standards?",
        "options": ["To save gel", "Extrapolation outside the calibrated range is unreliable", "Rings merge otherwise", "It makes wells larger"],
        "answer": 1,
        "why": "The relationship is only validated across the standards used. Re-run the sample at another dilution if it falls outside.",
    },
    {
        "topic": "SRID",
        "q": "A standard curve has R² = 0.91. What is the best action?",
        "options": ["Use it anyway", "Review ring measurements and repeat the standards", "Delete the lowest R² point silently", "Convert to log scale automatically"],
        "answer": 1,
        "why": "Low R² indicates measurement or assay problems. Investigate; never drop points without a documented reason.",
    },
    {
        "topic": "DID",
        "q": "Two precipitin lines that merge smoothly into one continuous arc indicate:",
        "options": ["Non-identity", "Partial identity", "Identity", "No reaction"],
        "answer": 2,
        "why": "A fused continuous line means both antigens share the same epitopes recognised by the antibody.",
    },
    {
        "topic": "DID",
        "q": "A fused line with a spur pointing past the junction indicates:",
        "options": ["Identity", "Partial identity", "Non-identity", "Prozone effect"],
        "answer": 1,
        "why": "The spur shows extra epitopes on one antigen that the other lacks – partial identity.",
    },
    {
        "topic": "DID",
        "q": "Crossing precipitin lines indicate:",
        "options": ["Identity", "Partial identity", "Non-identity", "Antibody deficiency"],
        "answer": 2,
        "why": "Each antibody population reacts independently with its own antigen, so the lines cross without interaction.",
    },
    {
        "topic": "DID",
        "q": "The precipitin line sits very close to the antibody well. What does that suggest?",
        "options": ["Antibody is in excess", "Antigen is in excess / antibody is limiting", "The gel is too thick", "Plate is contaminated"],
        "answer": 1,
        "why": "The line forms at equivalence; it shifts toward the well holding the less abundant (or slower diffusing) reactant.",
    },
    {
        "topic": "Fick's law",
        "q": "Fick's first law states that diffusion flux is:",
        "options": ["Proportional to concentration", "Proportional to the concentration gradient", "Independent of D", "Proportional to time"],
        "answer": 1,
        "why": "J = −D · dC/dx. Flux runs from high to low concentration at a rate set by D and the gradient.",
    },
    {
        "topic": "Fick's law",
        "q": "If time of diffusion is increased 4×, the characteristic diffusion distance √(Dt) increases by:",
        "options": ["2×", "4×", "8×", "16×"],
        "answer": 0,
        "why": "Distance scales with √t, so 4× the time gives only 2× the distance.",
    },
    {
        "topic": "Fick's law",
        "q": "Which change slows diffusion in the gel (lower D)?",
        "options": ["Higher temperature", "Larger molecule / higher viscosity", "Lower agarose concentration", "Smaller hydrodynamic radius"],
        "answer": 1,
        "why": "Stokes–Einstein: D = kT / (6πηr). Bigger r or higher η lowers D.",
    },
    {
        "topic": "General",
        "q": "Why are image-detected rings treated as 'candidates'?",
        "options": ["The software is random", "Edges can be artefacts; the researcher must verify boundaries", "Rings are always wrong", "Pixels cannot be measured"],
        "answer": 1,
        "why": "Automatic edge detection can pick bubbles, scratches or labels. Verification keeps results scientifically defensible.",
    },
]

GLOSSARY = {
    "Precipitin line": "Visible band where antigen and antibody meet in equivalence and form an insoluble lattice.",
    "Equivalence zone": "Antigen:antibody ratio giving maximum precipitation.",
    "Prozone": "Antibody excess reducing visible precipitate (false negative risk).",
    "Postzone": "Antigen excess dissolving the lattice and weakening the line.",
    "Mancini method": "End-point SRID analysis using D² versus concentration.",
    "Fahey–McKelvey": "Kinetic SRID analysis (before end-point) using D versus log C.",
    "Epitope": "Part of an antigen recognised by an antibody.",
    "Diffusion coefficient (D)": "Rate constant describing how fast a molecule spreads; units cm²/s.",
}

THEORY = {
    "SRID": """
**Single radial immunodiffusion** places antigen in a well cut into agarose that already contains specific antibody.
The antigen diffuses outward radially. Where antigen and antibody reach equivalence a circular precipitin ring forms.

- **End-point (Mancini):** ring area ∝ antigen amount, so **D² ∝ C**.
- **Kinetic (Fahey–McKelvey):** read before the ring stops growing; **D ∝ log C**.
- Always run ≥ 3 standards in the same plate as the unknowns.
""",
    "DID": """
**Double immunodiffusion (Ouchterlony)** places antigen and antibody in neighbouring wells. Both diffuse toward each other
and a precipitin line forms where they meet at equivalence.

| Pattern | What you see | Meaning |
|---|---|---|
| Identity | Lines fuse in a smooth arc | Same epitopes |
| Partial identity | Fused lines plus a spur | Shared and unique epitopes |
| Non-identity | Lines cross | Unrelated antigens |
""",
    "FICK": """
**Fick's first law:** J = −D · ∂C/∂x  (flux down a concentration gradient)

**Fick's second law:** ∂C/∂t = D · ∂²C/∂x²  (how the profile changes with time)

For a constant-concentration source in a semi-infinite gel: **C(x,t) = C₀ · erfc(x / 2√(Dt))**

The characteristic diffusion distance is **x ≈ 2√(Dt)**, which is why ring and line positions depend on time, not just
concentration.
""",
}
