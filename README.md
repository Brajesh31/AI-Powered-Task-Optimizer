# AI-Powered Task Optimizer

AI-Powered Task Optimizer is a Python prototype for emotion-aware productivity experiments. It combines text sentiment analysis, OpenCV face detection, microphone-related dependencies, task recommendations, and JSON mood history. Supporting scripts generate historical and team-level mood charts.

## Important safety and privacy notice

This is an educational prototype, not a medical, employment, psychological, or performance-evaluation system. Emotion inference is probabilistic and can be inaccurate or biased. Do not use it to make decisions about hiring, discipline, promotion, health, or access to services. Obtain consent, minimize collection, and use synthetic or de-identified data.

## Repository contents

- `main.py` - interactive camera, text sentiment, and task recommendation flow.
- `Historical Mood Tracking.py` - records and summarizes sample employee moods.
- `Team Mood Analytics.py` - aggregates `mood_tracking.json` and writes `team_mood_trends.png`.
- `Data Privacy Anonymization.py` - anonymization experiment.
- `Stress Management.py` - stress-related analysis experiment.
- `requirements.txt` - Python dependencies.
- `package.json` - legacy Node.js dependency metadata; no `index.js` is currently present.
- `mood_tracking.json`, `anonymized_data.csv`, and image/PDF/PPTX files - sample outputs or project artifacts.

## Setup

Use Python 3.10 or a compatible supported version:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

The main flow may require a working webcam, microphone permissions, graphical display, and model downloads. Press `q` to stop camera capture. The first Transformers run may download the pinned sentiment model.

## Mood tracking

`Historical Mood Tracking.py` writes timestamps and moods to `mood_tracking.json`. Run it from the repository root:

```powershell
python "Historical Mood Tracking.py"
python "Team Mood Analytics.py"
```

The analytics script currently contains a machine-specific absolute path and should be updated to use a repository-relative path before portable use.

## Development

Keep data collection opt-in and avoid storing raw camera frames, audio, text, or identifiable employee information. Add tests for unavailable devices, missing model files, empty mood data, unknown emotions, malformed JSON, and cleanup on interruption.

Run syntax checks with:

```powershell
python -m compileall .
```

See [docs/architecture.md](docs/architecture.md), [docs/development.md](docs/development.md), and [docs/security.md](docs/security.md).

## Ownership

Maintained by Brajesh Kumar (`brajesh31`).

Last updated: 2026-10-24 05:20 UTC
