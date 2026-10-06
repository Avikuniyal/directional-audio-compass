# tests/test_bearing.py  (owner: Avik). G1 now; B1 to B7 once dac/synth.py lands (Section 7.2).
import math
from dac import config


def test_g1_config_geometry():
    for row, phi_deg in zip(config.MIC_XY_M, config.MIC_BEARING_DEG):
        phi = math.radians(phi_deg)
        x = config.MIC_RADIUS_M * math.sin(phi)
        y = config.MIC_RADIUS_M * math.cos(phi)
        assert abs(row[0] - x) < 1e-5, ("x wrong for mic at", phi_deg)
        assert abs(row[1] - y) < 1e-5, ("y wrong for mic at", phi_deg)
