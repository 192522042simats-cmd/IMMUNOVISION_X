# IMMUNOVISION_X

A research support application for image-based analysis of
Single Radial Immunodiffusion (SRID) and Double Immunodiffusion (DID).

## Features
- SRID image processing using the existing analysis engine
- Candidate ring measurement review
- D² calculation
- Researcher-entered verification
- Calibration curve from supplied antigen standards
- DID image processing and researcher interpretation
- CSV export

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Scientific limitations
Automatic image detections are candidates and require researcher review.
Concentration estimation requires valid experimental standards.
Pixel measurements require scale calibration before conversion to physical units.
DID pattern interpretation requires visual confirmation.

## Deployment
Deploy this repository using Streamlit Community Cloud.
Set the main file path to `streamlit_app.py`.
