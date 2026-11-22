IMMUNOVISION_X
Analysis and teaching workspace for Single Radial Immunodiffusion (SRID) and
Double Immunodiffusion (DID).
Files
File	Purpose
`streamlit_app.py`	The web app (all pages)
`analysis_engine.py`	Image detection of wells, rings and lines
`science.py`	Calibration, concentration estimate, Fick's laws
`content.py`	Theory text, glossary, quiz questions
`demo.py`	Synthetic teaching plates
`.streamlit/config.toml`	Theme
Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
Deploy
Streamlit Community Cloud, main file `streamlit_app.py`. Commit every file
above, including the `.streamlit` folder.
Scientific limitations
Detections are candidates and need researcher verification. Concentration
needs at least three standards on the same plate. Pixel values need a known
well diameter for mm. DID interpretation is always the researcher's.
Synthetic teaching plates must never be mixed with experimental data.
