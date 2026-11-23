"""loop-asr: plain vs looped shared-block CTC encoders for tiny English STT."""

from loop_asr.config import Config, ModelConfig, load_config
from loop_asr.device import get_device
from loop_asr.flops import forward_flops
from loop_asr.model import (
    CTCModel,
    LoopedEncoder,
    PlainEncoder,
    build_model,
    count_params,
    ctc_feasible,
)

__all__ = [
    "CTCModel",
    "Config",
    "LoopedEncoder",
    "ModelConfig",
    "PlainEncoder",
    "build_model",
    "count_params",
    "ctc_feasible",
    "forward_flops",
    "get_device",
    "load_config",
]
