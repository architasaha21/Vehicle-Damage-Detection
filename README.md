# VisionDamage AI — Streamlit App

## Setup

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate it
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your trained model
#    Copy best.pt from Google Drive into the model/ folder
mkdir model
# → paste best.pt here

# 5. Run the app
streamlit run app.py
```

## Project Structure

```
vehicle_damage_app/
├── app.py                  ← Streamlit frontend
├── requirements.txt
├── model/
│   └── best.pt             ← your trained YOLOv8 weights
├── utils/
│   └── detector.py         ← inference + Grad-CAM logic
└── sample_images/          ← optional test images
```

## Features
- Upload vehicle image → real-time bounding box detection
- Grad-CAM heatmap overlay
- Severity gauge (0–100)
- Per-damage breakdown (type, severity, impact, confidence)
- Downloadable JSON report
