"""T-Rex slow/fast policy boundary specialized for one Revo 3 hand."""

from .adapter import (
    CallableTReXBackend,
    MockTReXBackend,
    TReXBackend,
    TReXRevoPolicyAdapter,
)
from .aggregation import (
    ActionTemporalAggregator,
    TemporalAggregationError,
)
from .cache import CacheProtocolError, SlowFastCache
from .checkpoint import (
    CheckpointMigrationReport,
    build_revo_trex_model,
    load_revo_compatible_state_dict,
)
from .contracts import (
    ACTION_CHUNK,
    ACTION_DIM,
    ActionChunk,
    InferenceMode,
    PolicyObservation,
    PolicyRequest,
    TaskKey,
)
from .dataset import (
    RevoEpisode,
    RevoEpisodeAdapter,
    RevoNormStats,
    RevoTrainingSample,
    build_revo_feature_schema,
)
from .schedule import MAIN_ALIGNED_SCHEDULE, PolicySchedule
from .runner import TReXPolicyRunner
from .zmq_backend import (
    RequestReplyTransport,
    TReXTransportError,
    TReXTransportTimeout,
    TReXWireProtocolError,
    TReXZmqError,
    ZmqReqTransport,
    ZmqTReXBackend,
)

__all__ = [
    "ACTION_CHUNK",
    "ACTION_DIM",
    "ActionChunk",
    "ActionTemporalAggregator",
    "CacheProtocolError",
    "CallableTReXBackend",
    "CheckpointMigrationReport",
    "InferenceMode",
    "MAIN_ALIGNED_SCHEDULE",
    "MockTReXBackend",
    "PolicyObservation",
    "PolicyRequest",
    "PolicySchedule",
    "RevoEpisode",
    "RevoEpisodeAdapter",
    "RevoNormStats",
    "RevoTrainingSample",
    "RequestReplyTransport",
    "SlowFastCache",
    "TReXBackend",
    "TReXTransportError",
    "TReXTransportTimeout",
    "TReXRevoPolicyAdapter",
    "TReXPolicyRunner",
    "TReXWireProtocolError",
    "TReXZmqError",
    "TaskKey",
    "TemporalAggregationError",
    "build_revo_trex_model",
    "build_revo_feature_schema",
    "load_revo_compatible_state_dict",
    "ZmqReqTransport",
    "ZmqTReXBackend",
]
