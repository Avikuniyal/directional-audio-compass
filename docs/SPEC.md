# Spec

The full spec is `Directional Audio Compass Technical Spec and Sprint Plan.md` in the repo root.

One-page summary: a ReSpeaker XVF3800 4-mic array at 16 kHz feeds 1024-sample blocks. GCC-PHAT
per mic pair (dac/bearing.py) and SRP-PHAT over a 2 degree grid (dac/srp.py) give a bearing and
confidence. The tracker (dac/tracker.py) turns bearings and loudness into an approach flag. A
YAMNet-based classifier (dac/classifier.py) gates the alert to vehicle sounds. The pipeline and
websocket server feed a compass frontend (app/index.html).
