"""T-Rex slow/fast policy boundary specialized for one Revo 3 hand."""

from .adapter import (
    CallableTReXBackend,
    MockTReXBackend,
    TReXBackend,
    TReXRevoPolicyAdapter,
)
from .async_runner import AsyncPolicyPoll, AsyncPolicyState, AsyncTReXPolicyRunner
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
from .artifacts import (
    sha256_file,
    validate_revo_deform_artifact,
    validate_revo_vqvae_artifact,
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
from .training import (
    RevoStageSpec,
    RevoTrainingStage,
    build_revo_optimizer_groups,
    configure_revo_trainable_parameters,
    stage_spec,
)
from .tactile_profile import (
    TactileProfile,
    TactileProfileSpec,
    tactile_profile_spec,
)
from .runner import TReXPolicyRunner
from .server_identity import (
    SERVER_IDENTITY_SCHEMA,
    TReXServerIdentity,
    build_revo_server_identity,
)
from .zmq_backend import (
    OFFICIAL_SINGLE_VIEW_PROFILE,
    REVO3_FULL_CENTER_PROFILE,
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
    "AsyncPolicyPoll",
    "AsyncPolicyState",
    "AsyncTReXPolicyRunner",
    "ActionChunk",
    "ActionTemporalAggregator",
    "CacheProtocolError",
    "CallableTReXBackend",
    "CheckpointMigrationReport",
    "InferenceMode",
    "MAIN_ALIGNED_SCHEDULE",
    "MockTReXBackend",
    "OFFICIAL_SINGLE_VIEW_PROFILE",
    "PolicyObservation",
    "PolicyRequest",
    "PolicySchedule",
    "RevoEpisode",
    "RevoEpisodeAdapter",
    "RevoNormStats",
    "RevoTrainingSample",
    "REVO3_FULL_CENTER_PROFILE",
    "SERVER_IDENTITY_SCHEMA",
    "RevoStageSpec",
    "RevoTrainingStage",
    "RequestReplyTransport",
    "SlowFastCache",
    "TReXBackend",
    "TReXTransportError",
    "TReXTransportTimeout",
    "TReXRevoPolicyAdapter",
    "TReXPolicyRunner",
    "TReXServerIdentity",
    "TReXWireProtocolError",
    "TReXZmqError",
    "TaskKey",
    "TactileProfile",
    "TactileProfileSpec",
    "TemporalAggregationError",
    "build_revo_trex_model",
    "build_revo_server_identity",
    "build_revo_feature_schema",
    "build_revo_optimizer_groups",
    "configure_revo_trainable_parameters",
    "load_revo_compatible_state_dict",
    "sha256_file",
    "stage_spec",
    "tactile_profile_spec",
    "validate_revo_deform_artifact",
    "validate_revo_vqvae_artifact",
    "ZmqReqTransport",
    "ZmqTReXBackend",
]
