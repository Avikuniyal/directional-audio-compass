from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class BearingResult:
    bearing_deg: Optional[float]   # 0 <= x < 360, clockwise from front. None if the block is silent.
    confidence: float              # 0.0 to 1.0 (Section 5.4)
    srp: np.ndarray                # shape (180,), P(theta_k) on ANGLE_GRID_DEG


@dataclass
class TrackState:
    t_s: float                            # time of the block's first sample
    bearing_deg: Optional[float]          # latest accepted bearing, None if none accepted
    bearing_smooth_deg: Optional[float]   # circular mean over DISPLAY_WINDOW_S, for the display
    confidence: float                     # confidence of this block's bearing
    level_dbfs: float                     # this block's loudness (Section 5.5)
    trend_dbps: Optional[float]           # loudness slope, None until the window has filled
    rate_dps: Optional[float]             # bearing slope, None if too few accepted bearings
    noise_floor_dbfs: float               # 10th percentile of level over FLOOR_WINDOW_S
    approach: bool                        # approach flag with hysteresis (Section 5.6)


@dataclass
class ClassState:
    t_s: float
    label: str            # 'vehicle' or 'other'
    prob_vehicle: float   # 0.0 to 1.0, latest window
    present: bool         # prob_vehicle >= CLASS_THRESH within the last CLASS_HOLD_S seconds
