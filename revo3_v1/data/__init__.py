"""Validated Revo3 single-hand episode and T-Rex training-data adapters."""

from .episode import RevoEpisode, RevoEpisodeMeta
from .synthetic import SyntheticRevoConfig, generate_synthetic_revo_episodes
from .trex_json import ConversionConfig, convert_revo_episodes_to_trex_json

__all__ = [
    "ConversionConfig",
    "RevoEpisode",
    "RevoEpisodeMeta",
    "SyntheticRevoConfig",
    "convert_revo_episodes_to_trex_json",
    "generate_synthetic_revo_episodes",
]
