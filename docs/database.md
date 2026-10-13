# Data storage

The prototype stores mood history in `mood_tracking.json` as employee IDs mapped to timestamped mood entries. This is a local demonstration format, not a secure production datastore.

Do not use real employee identifiers. A production system would need ownership checks, encryption, retention and deletion policies, audit records, and schema migrations.
