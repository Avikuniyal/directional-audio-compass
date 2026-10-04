# dac/config.py
# Single source of truth for every constant. Never hard-code these elsewhere.

# ---- Geometry and signal chain (owner: Avik, LOCKED) ----
FS_HZ = 16000
C_MPS = 343.0
N_MICS = 4
N_RAW_CHANNELS = 6
CHANNEL_ORDER = [4, 3, 2, 5]      # raw channel feeding columns 0, 1, 2, 3
MIC_BEARING_DEG = [45.0, 135.0, 225.0, 315.0]
MIC_RADIUS_M = 0.0470
MIC_XY_M = [                      # (x right, y front) in meters, rows = columns
    [0.03323, 0.03323],
    [0.03323, -0.03323],
    [-0.03323, -0.03323],
    [-0.03323, 0.03323],
]
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
BLOCK_N = 1024
HOP_N = 512
DEVICE_NAME_SUBSTRING = 'XVF3800'

# ---- Bearing (owner: Avik, PROVISIONAL) ----
N_FFT = 2048
UPSAMPLE = 16
BAND_HZ = (200.0, 7000.0)
ANGLE_GRID_DEG = list(range(0, 360, 2))   # 180 candidates
MIN_LEVEL_DBFS = -60.0

# ---- Tracker (owner: Srihaas, PROVISIONAL) ----
CONF_MIN = 0.3
TREND_WINDOW_S = 1.0
RATE_WINDOW_S = 1.0
RATE_MIN_POINTS = 8
DISPLAY_WINDOW_S = 0.25
RATE_MAX_DPS = 15.0
TREND_MIN_DBPS = 3.0
LEVEL_MARGIN_DB = 10.0
FLOOR_WINDOW_S = 10.0
ON_HOLD_S = 0.5
OFF_HOLD_S = 1.5

# ---- Classifier, fusion, app (owner: Anirudh, PROVISIONAL) ----
CLASS_CHANNEL = 0                 # column 0, the 45 degree mic
CLASS_WINDOW_S = 0.96
CLASS_THRESH = 0.5
CLASS_HOLD_S = 3.0
ALERT_ENABLED = True              # False puts the app in fallback level L1
WS_HOST = 'localhost'
WS_PORT = 8765
SEND_RATE_HZ = 20
