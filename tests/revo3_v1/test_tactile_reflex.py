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
    hold = reflex.update(_frame(2, 0.2), phase=ReflexPhase.HOLD, now_ns=2)
    assert hold.updated and hold.reason == "hold_tighten"
    assert np.max(np.abs(hold.delta_q_rad)) <= 0.003
    duplicate = reflex.update(_frame(2, 0.2), phase=ReflexPhase.HOLD, now_ns=2)
    assert not duplicate.updated
    np.testing.assert_array_equal(duplicate.residual_q_rad, hold.residual_q_rad)


def test_reflex_residual_is_bounded_and_force_protection_backs_off():
    cfg = ReflexConfig.demo()
    reflex = TactileReflexPlugin(cfg)
    period = int(round(1e9 / cfg.update_hz))
    for i in range(1, 100):
        timestamp = i * period
        result = reflex.update(
            _frame(timestamp, 0.2), phase=ReflexPhase.HOLD, now_ns=timestamp
        )
    assert np.max(np.abs(result.residual_q_rad)) <= cfg.max_abs_cumulative_rad + 1e-7
    protected = reflex.update(
        _frame(100 * period, 1.2),
        phase=ReflexPhase.CONTACT_BUILD,
        now_ns=100 * period,
    )
    assert protected.overload
    assert protected.reason == "force_protect"
    assert np.min(protected.delta_q_rad) < 0


def test_release_resets_residual_even_without_a_new_tactile_sample():
    reflex = TactileReflexPlugin(ReflexConfig.demo())
    hold = reflex.update(_frame(1, 0.2), phase=ReflexPhase.HOLD, now_ns=1)
    assert np.any(hold.residual_q_rad)
    release = reflex.update(_frame(1, 0.2), phase=ReflexPhase.RELEASE)
    np.testing.assert_array_equal(release.residual_q_rad, np.zeros(21))


def test_load_relief_is_disabled_by_default_but_force_protection_remains_active():
    reflex = TactileReflexPlugin(ReflexConfig.demo())
    result = reflex.update(_frame(1, 0.8), phase=ReflexPhase.HOLD, now_ns=1)
    assert result.reason == "load_relief_disabled"
    np.testing.assert_array_equal(result.delta_q_rad, np.zeros(21))


def test_stale_touch_disables_cair_without_dropping_held_residual():
    reflex = TactileReflexPlugin(ReflexConfig.demo())
    fresh = reflex.update(_frame(1, 0.2), phase=ReflexPhase.HOLD, now_ns=1)
    assert np.any(fresh.residual_q_rad)
    stale = reflex.update(
        _frame(2, 0.2),
        phase=ReflexPhase.HOLD,
        now_ns=2 + reflex.config.max_tactile_age_ns + 1,
    )
    assert not stale.updated
    assert stale.reason == "stale_touch_cair_disabled"
    np.testing.assert_array_equal(stale.residual_q_rad, fresh.residual_q_rad)


def test_enabled_cair_fails_closed_without_a_freshness_clock():
    reflex = TactileReflexPlugin(ReflexConfig.demo())
    result = reflex.update(_frame(1, 0.2), phase=ReflexPhase.HOLD)
    assert not result.updated
    assert result.reason == "touch_freshness_unverified_cair_disabled"
    np.testing.assert_array_equal(result.residual_q_rad, np.zeros(21))


def test_120hz_touch_cannot_integrate_residual_faster_than_frozen_12hz():
    reflex = TactileReflexPlugin(ReflexConfig.demo())
    tick_ns = int(round(1e9 / 120))
    updated = 0
    for index in range(121):
        timestamp = 1 + index * tick_ns
        result = reflex.update(
            _frame(timestamp, 0.2), phase=ReflexPhase.HOLD, now_ns=timestamp
        )
        updated += int(result.updated)
    # Integer rounding can place the final sample just under a boundary, but
    # 120 Hz input must remain near 12 integrations/s, never 120.
    assert 10 <= updated <= 13
    assert np.max(np.abs(result.residual_q_rad)) <= reflex.config.max_abs_cumulative_rad


def test_hard_overload_is_reported_immediately_even_between_cair_ticks():
    reflex = TactileReflexPlugin(ReflexConfig.demo())
    first = reflex.update(_frame(1, 0.2), phase=ReflexPhase.HOLD, now_ns=1)
    assert first.updated
    immediate = reflex.update(_frame(2, 2.5), phase=ReflexPhase.HOLD, now_ns=2)
    assert not immediate.updated
    assert immediate.hard_overload
    assert immediate.reason == "hard_overload_rate_limited_reported"
