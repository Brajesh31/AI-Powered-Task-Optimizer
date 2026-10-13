# API

No stable network API is currently defined. The primary interfaces are Python classes and scripts:

- `EmotionAnalyzer` analyzes text and detects faces.
- `TaskRecommender` maps supported emotion labels to task suggestions.
- `record_mood` and `analyze_mood_trends` manage local JSON history.

Future APIs must define authentication, input validation, confidence semantics, privacy, and error responses.
