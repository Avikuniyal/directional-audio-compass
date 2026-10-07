# Directional Audio Compass

Real-time directional sound alerts for deaf and hard-of-hearing users, using a
4-microphone array and our own time-difference-of-arrival processing
(GCC-PHAT and SRP-PHAT), with a trained classifier that gates alerts to vehicle sounds.

## Team
- Avik: GCC-PHAT and capture (dac/bearing.py, dac/capture.py)
- Srihaas: SRP-PHAT, tracking and approach detection (dac/srp.py, dac/tracker.py, dac/synth.py)
- Anirudh: classifier, integration, frontend (dac/classifier.py, dac/pipeline.py, app/)

## Hardware
ReSpeaker XVF3800 USB 4-Mic Array, 6-channel raw firmware v2.0.8, 16 kHz.
See docs/CHANNEL_MAP.md for geometry and channel order.

## Run it
pip install -r requirements.txt
python -m dac.server --source live
Then open app/index.html in a browser.

## Test it
pytest

## How it works
(Plain-language explanation from the submission kit.)

## Credits and data
YAMNet (Google, TensorFlow Hub). UrbanSound8K. ESC-50. Sound clips: see data/manifest.csv.

## AI use
Summary of docs/AI_USE.md.
