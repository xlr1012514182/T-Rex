import numpy as np

from revo3_v1.tactile import (
    ReflexConfig,
    ReflexPhase,
    TactileFrame,
    TactileReflexPlugin,
)


def _frame(timestamp, normal_force):
    f6 = np.zeros((5, 6), dtype=np.float32)
    f6[:, 2] = normal_force
    return TactileFrame(timestamp, f6, sequence=timestamp)


def test_reflex_is_zero_before_contact_and_integrates_only_new_samples():
    reflex = TactileReflexPlugin(ReflexConfig.demo())
    first = reflex.update(_frame(1, 0.2), phase=ReflexPhase.PRECONTACT)
    np.testing.assert_array_equal(first.residual_q_rad, np.zeros(21))
    hold = reflex.update(_frame(2, 0.2), phase=ReflexPhase.HOLD)
    assert hold.updated and hold.reason == "hold_tighten"
    assert np.max(np.abs(hold.delta_q_rad)) <= 0.003
    duplicate = reflex.update(_frame(2, 0.2), phase=ReflexPhase.HOLD)
    assert not duplicate.updated
    np.testing.assert_array_equal(duplicate.residual_q_rad, hold.residual_q_rad)


def test_reflex_residual_is_bounded_and_force_protection_backs_off():
    cfg = ReflexConfig.demo()
    reflex = TactileReflexPlugin(cfg)
    for i in range(1, 100):
        result = reflex.update(_frame(i, 0.2), phase=ReflexPhase.HOLD)
    assert np.max(np.abs(result.residual_q_rad)) <= cfg.max_abs_cumulative_rad + 1e-7
    protected = reflex.update(_frame(100, 1.2), phase=ReflexPhase.CONTACT_BUILD)
    assert protected.overload
    assert protected.reason == "force_protect"
    assert np.min(protected.delta_q_rad) < 0


def test_release_resets_residual_even_without_a_new_tactile_sample():
    reflex = TactileReflexPlugin(ReflexConfig.demo())
    hold = reflex.update(_frame(1, 0.2), phase=ReflexPhase.HOLD)
    assert np.any(hold.residual_q_rad)
    release = reflex.update(_frame(1, 0.2), phase=ReflexPhase.RELEASE)
    np.testing.assert_array_equal(release.residual_q_rad, np.zeros(21))
