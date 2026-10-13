# Troubleshooting

## Model download or import failure

Confirm the virtual environment and dependency installation. The first Transformers run may download a model.

## Camera failure

Check camera permissions, device availability, and OpenCV GUI support. The application should release the capture device when interrupted.

## Analytics path failure

Run from the repository root and replace machine-specific absolute paths with portable paths.

## Empty mood data

Check that `mood_tracking.json` exists and contains valid JSON records before running aggregation scripts.
