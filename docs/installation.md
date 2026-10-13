# Installation

Install Python 3.10, create a virtual environment, and install:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The main program may need a webcam, microphone permissions, OpenCV GUI support, and internet access for the initial model download. Use synthetic data when hardware is unavailable.
