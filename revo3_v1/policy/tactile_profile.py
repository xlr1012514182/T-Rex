"""Capability-bound tactile model profiles.

Profiles are checkpoint families, not runtime switches.  A checkpoint trained
for Force6D+DIFF cannot be served as pressure/matrix or DIFF-only by changing a
flag.  Unsupported profile C is represented explicitly so the launcher fails
instead of reshaping pressure values into fake Force6D.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class TactileProfile(str, Enum):
    PROFILE_A_FORCE6D_DIFF = "profile_a_force6d_diff"
    PROFILE_B_DIFF_ONLY = "profile_b_diff_only"
    PROFILE_C_PRESSURE_MATRIX = "profile_c_pressure_matrix"
    ABLATION_FORCE6D_ONLY = "ablation_force6d_only"


@dataclass(frozen=True)
class TactileProfileSpec:
    profile: TactileProfile
    use_tactile_vec: int
    use_tactile_deform: int
    use_tactile_vqvae: int
    use_tactile_code: int
    history_shape: tuple[int, ...] | None
    deform_shape: tuple[int, ...] | None
    launchable_by_current_trainer: bool
    ablation_only: bool = False


TACTILE_PROFILE_SPECS: Mapping[TactileProfile, TactileProfileSpec] = {
    TactileProfile.PROFILE_A_FORCE6D_DIFF: TactileProfileSpec(
        profile=TactileProfile.PROFILE_A_FORCE6D_DIFF,
        use_tactile_vec=1,
        use_tactile_deform=1,
        use_tactile_vqvae=1,
        use_tactile_code=1,
        history_shape=(16, 5, 6),
        deform_shape=(5, 1, 240, 240),
        launchable_by_current_trainer=True,
    ),
    TactileProfile.PROFILE_B_DIFF_ONLY: TactileProfileSpec(
        profile=TactileProfile.PROFILE_B_DIFF_ONLY,
        use_tactile_vec=0,
        use_tactile_deform=1,
        use_tactile_vqvae=0,
        use_tactile_code=0,
        history_shape=None,
        deform_shape=(5, 1, 240, 240),
        launchable_by_current_trainer=True,
    ),
    TactileProfile.PROFILE_C_PRESSURE_MATRIX: TactileProfileSpec(
        profile=TactileProfile.PROFILE_C_PRESSURE_MATRIX,
        use_tactile_vec=0,
        use_tactile_deform=0,
        use_tactile_vqvae=0,
        use_tactile_code=0,
        history_shape=None,
        deform_shape=None,
        launchable_by_current_trainer=False,
    ),
    TactileProfile.ABLATION_FORCE6D_ONLY: TactileProfileSpec(
        profile=TactileProfile.ABLATION_FORCE6D_ONLY,
        use_tactile_vec=1,
        use_tactile_deform=0,
        use_tactile_vqvae=0,
        use_tactile_code=0,
        history_shape=(16, 5, 6),
        deform_shape=None,
        launchable_by_current_trainer=True,
        ablation_only=True,
    ),
}


def tactile_profile_spec(profile: TactileProfile | str) -> TactileProfileSpec:
    return TACTILE_PROFILE_SPECS[TactileProfile(profile)]
