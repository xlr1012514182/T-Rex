"""Validated Revo3 single-hand episode and T-Rex training-data adapters."""

from .episode import RevoEpisode, RevoEpisodeMeta
from .native_touch import NativeForceStream, native_force_stream
from .synthetic import SyntheticRevoConfig, generate_synthetic_revo_episodes
from .splits import CorpusSplit, RevoCorpusSplitManifest, SplitEntry
from .trex_json import ConversionConfig, convert_revo_episodes_to_trex_json

__all__ = [
    "ConversionConfig",
    "CorpusSplit",
    "RevoEpisode",
    "RevoEpisodeMeta",
    "NativeForceStream",
    "RevoCorpusSplitManifest",
    "SplitEntry",
    "SyntheticRevoConfig",
    "convert_revo_episodes_to_trex_json",
    "generate_synthetic_revo_episodes",
    "native_force_stream",
]
