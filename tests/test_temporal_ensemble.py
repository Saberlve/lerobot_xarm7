import logging
import math
import time

import numpy as np
import pytest

from lerobot_robot_ufactory.utils.starvla_ws_client.temporal_ensemble import (
    TemporalEnsembleClientWrapper,
    TemporalEnsembler,
)


# ---------------------------------------------------------------------------
# TemporalEnsembler
# ---------------------------------------------------------------------------
def test_weighted_average_matches_manual():
    # coeff=0.5 -> weights 1 and exp(-0.5) for the first and second chunk
    # merged into a step.
    ens = TemporalEnsembler(action_horizon=4, coeff=0.5)
    ens.insert(np.zeros((4, 2)), start_step=0)
    assert np.allclose(ens.get_action(0), [0.0, 0.0])  # only chunk0 covers t=0

    ens.insert(np.ones((4, 2)), start_step=1)
    w1 = math.exp(-0.5)
    expected = w1 / (1 + w1)  # step 1: chunk0 (w=1) + chunk1 (w=exp(-0.5))
    assert np.allclose(ens.get_action(1), [expected, expected], atol=1e-6)
    # Positive coeff favors the OLDER chunk (lerobot convention).
    assert expected < 0.5


def test_matches_lerobot_act_temporal_ensembler():
    """With query_period=1 and zero latency the numpy ensembler must reproduce
    lerobot's ACTTemporalEnsembler step for step."""
    torch = pytest.importorskip("torch")
    from lerobot.policies.act.modeling_act import ACTTemporalEnsembler

    rng = np.random.default_rng(0)
    chunk_size, steps, dim, coeff = 6, 6, 3, 0.01
    chunks = rng.normal(size=(steps, chunk_size, dim)).astype(np.float32)

    ref = ACTTemporalEnsembler(coeff, chunk_size)
    ens = TemporalEnsembler(action_horizon=chunk_size, coeff=coeff)
    for t in range(steps):
        ref_action = ref.update(torch.from_numpy(chunks[t][None])).numpy()[0]
        ens.insert(chunks[t], start_step=t)
        assert np.allclose(ens.get_action(t), ref_action, atol=1e-4)


def test_gripper_taken_from_newest_chunk_when_disabled():
    ens = TemporalEnsembler(action_horizon=4, coeff=0.5, ensemble_gripper=False)
    chunk0 = np.zeros((4, 2))
    chunk1 = np.ones((4, 2))
    ens.insert(chunk0, start_step=0)
    ens.insert(chunk1, start_step=1)
    action = ens.get_action(1)
    # Joint dim is averaged (both chunks agree on 0 vs 1 -> blended), but the
    # gripper dim must be the newest chunk's value, not the blend.
    assert action[0] == pytest.approx(math.exp(-0.5) / (1 + math.exp(-0.5)), abs=1e-6)
    assert action[-1] == pytest.approx(1.0)


def test_no_coverage_raises():
    ens = TemporalEnsembler(action_horizon=4, coeff=0.01)
    with pytest.raises(RuntimeError):
        ens.get_action(0)
    ens.insert(np.zeros((4, 2)), start_step=0)
    with pytest.raises(RuntimeError):
        ens.get_action(4)  # chunk covers steps 0..3 only


def test_late_chunk_drops_consumed_steps():
    ens = TemporalEnsembler(action_horizon=4, coeff=0.0)  # uniform weights
    ens.insert(np.zeros((4, 1)), start_step=0)
    for t in range(3):
        ens.get_action(t)
    # A chunk issued at t=0 arrives at t=3: its first 3 steps are dead.
    ens.insert(np.ones((4, 1)), start_step=0)
    # Step 3 is covered by both chunks -> uniform average 0.5.
    assert ens.get_action(3) == pytest.approx([0.5])


# ---------------------------------------------------------------------------
# TemporalEnsembleClientWrapper
# ---------------------------------------------------------------------------
class FakeClient:
    """predict_action returns fixed (T, D) chunks and records every call."""

    def __init__(self, chunks, delay_s=0.0):
        self._chunks = [np.asarray(c, dtype=np.float32) for c in chunks]
        self._delay_s = delay_s
        self.calls = []
        self.reset_count = 0

    def predict_action(self, query_info):
        self.calls.append(query_info)
        if self._delay_s:
            time.sleep(self._delay_s)
        i = min(len(self.calls) - 1, len(self._chunks) - 1)
        return {"data": {"actions": [self._chunks[i]]}}

    def reset(self):
        self.reset_count += 1


def _wait_chunk_index(wrapper, target, timeout_s=2.0):
    deadline = time.perf_counter() + timeout_s
    while wrapper.chunk_index < target:
        assert time.perf_counter() < deadline, "timed out waiting for chunk"
        wrapper._collect_pending(block=True)


def test_first_call_blocking_then_prefetch_at_period():
    client = FakeClient([np.zeros((8, 4)), np.ones((8, 4))])
    wrapper = TemporalEnsembleClientWrapper(client, query_period=3, action_horizon=8, coeff=0.01)

    actions = [wrapper.get_action({"t": t}) for t in range(3)]
    assert len(client.calls) == 1  # only the blocking first inference
    assert np.allclose(actions[0], 0.0)

    # t=3 hits the query period: a background inference fires with the query
    # captured at that tick, and the main thread keeps going.
    wrapper.get_action({"t": 3})
    assert len(client.calls) == 2
    assert client.calls[1]["t"] == 3

    _wait_chunk_index(wrapper, target=1)
    # Step 4 is covered by chunk0 (start 0) and chunk1 (start 3); the new
    # chunk enters with weight exp(-coeff).
    action = wrapper.get_action({"t": 4})
    w1 = math.exp(-0.01)
    assert np.allclose(action, w1 / (1 + w1), atol=1e-5)


def test_reset_clears_state_and_client():
    client = FakeClient([np.zeros((8, 4))])
    wrapper = TemporalEnsembleClientWrapper(client, query_period=2, action_horizon=8)
    wrapper.get_action({"t": 0})
    wrapper.reset()
    assert wrapper.chunk_index == -1
    assert client.reset_count == 1
    # First call after reset blocks on a fresh inference again.
    action = wrapper.get_action({"t": 0})
    assert len(client.calls) == 2
    assert action.shape == (4,)


def test_stale_chunk_dropped_with_warning(caplog):
    client = FakeClient([np.zeros((4, 4))])
    wrapper = TemporalEnsembleClientWrapper(client, query_period=1, action_horizon=4)
    wrapper._t = 10  # pretend 10 steps were already executed
    with caplog.at_level(logging.WARNING):
        wrapper._accept(np.zeros((4, 4)), start_step=1, elapsed_ms=1.0)
    assert wrapper.chunk_index == -1  # not accepted
    assert any("dropping stale chunk" in r.message for r in caplog.records)


def test_slow_inference_still_returns_actions():
    # Inference latency (~5 ticks of wall-clock work simulated by sleep) never
    # blocks the control loop: every get_action returns immediately with the
    # ensemble built so far.
    client = FakeClient([np.zeros((16, 4)), np.ones((16, 4))], delay_s=0.1)
    wrapper = TemporalEnsembleClientWrapper(client, query_period=2, action_horizon=16, coeff=0.01)
    actions = [wrapper.get_action({"t": t}) for t in range(10)]
    assert all(a.shape == (4,) for a in actions)
    _wait_chunk_index(wrapper, target=1)
    # Once the delayed chunk lands, it must be aligned to its issue step
    # (t=2), so a step after that blends old and new chunks.
    action = wrapper.get_action({"t": 11})
    assert 0.0 < action[0] < 1.0
