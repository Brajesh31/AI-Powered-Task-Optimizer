# Architecture

The prototype has four logical layers:

1. Device and text input through OpenCV, sound-related libraries, and terminal prompts.
2. Inference through Transformers sentiment analysis and experimental computer-vision/audio code.
3. Recommendation through the `TaskRecommender` mapping in `main.py`.
4. Persistence and analytics through `mood_tracking.json`, pandas, and matplotlib scripts.

Future changes should separate these layers behind explicit interfaces and avoid coupling raw personal data to model output.
