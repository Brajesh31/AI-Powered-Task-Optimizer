# Development

Run syntax checks with `python -m compileall .`. Test scripts from the repository root so relative files resolve correctly.

Known development concerns include the absolute path in `Historical Mood Tracking.py`, side effects from sample code executed at import time, random facial emotion labels, and device cleanup. Changes should make these behaviors explicit and testable.
