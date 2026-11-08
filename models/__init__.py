"""RAMoE and its original IMU and skeleton encoders."""
from .ramoe import ModalitySelector, PRISM
from .imu import TCN, TemporalConvEncoder
from .skeleton import STGCN_Encoder
from .unimodal import UniModalityModel

RAMoE = PRISM

__all__ = ["RAMoE", "PRISM", "ModalitySelector", "TCN", "TemporalConvEncoder",
           "STGCN_Encoder", "UniModalityModel"]
