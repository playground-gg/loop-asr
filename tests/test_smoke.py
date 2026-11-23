"""CPU smoke tests: forward, CTC loss, one optimizer step, loop count, closed gate at init,
param and FLOP matching between arms, early exits, CTC feasibility and the device helper."""

from pathlib import Path

import pytest
import torch

from loop_asr import (
    ModelConfig,
    build_model,
    count_params,
    ctc_feasible,
    forward_flops,
    get_device,
    load_config,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def tiny(kind: str) -> ModelConfig:
    return ModelConfig(
        kind=kind,
        n_mels=80,
        d_model=32,
        n_heads=2,
        d_ff=64,
        conv_channels=8,
        vocab_size=16,
        n_blocks=2,
        n_prelude=1,
        n_core=1,
        n_coda=1,
        gate_rank=4,
        default_loops=2,
    )


def batch(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(2, 120, 80, generator=g)  # fbank-like: (B, T frames, mels)
    lengths = torch.tensor([120, 96])
    targets = torch.randint(0, 16, (2, 8), generator=g)
    target_lengths = torch.tensor([8, 5])
    return feats, lengths, targets, target_lengths


@pytest.mark.parametrize("kind", ["plain", "looped"])
def test_forward_loss_and_step(kind):
    torch.manual_seed(0)
    model = build_model(tiny(kind))
    feats, lengths, targets, target_lengths = batch()
    log_probs, out_lengths = model(feats, lengths)
    assert log_probs.shape == (2, 29, 17)
    assert out_lengths.tolist() == [29, 23]
    loss = model.ctc_loss(log_probs, out_lengths, targets, target_lengths)
    assert torch.isfinite(loss)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = [p.detach().clone() for p in model.parameters()]
    opt.zero_grad()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    opt.step()
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters(), strict=True))
    loss2 = model.ctc_loss(*model(feats, lengths), targets, target_lengths)
    assert torch.isfinite(loss2)


@pytest.mark.parametrize("loops", [1, 3])
def test_looped_runs_at_loop_counts(loops):
    torch.manual_seed(0)
    model = build_model(tiny("looped"))
    feats, lengths, targets, target_lengths = batch()
    log_probs, out_lengths = model(feats, lengths, loops=loops)
    assert log_probs.shape == (2, 29, 17)
    loss = model.ctc_loss(log_probs, out_lengths, targets, target_lengths)
    assert torch.isfinite(loss)
    loss.backward()


def test_closed_gate_at_init_equals_no_loop_path():
    torch.manual_seed(0)
    model = build_model(tiny("looped")).eval()
    feats, lengths, *_ = batch()
    with torch.no_grad():
        one, _ = model(feats, lengths, loops=1)
        three, _ = model(feats, lengths, loops=3)
    torch.testing.assert_close(three, one, rtol=1e-5, atol=1e-6)


def test_open_gate_makes_loops_matter():
    torch.manual_seed(0)
    model = build_model(tiny("looped")).eval()
    for gate in model.encoder.gates:
        torch.nn.init.normal_(gate.up.weight, std=0.5)
    feats, lengths, *_ = batch()
    with torch.no_grad():
        one, _ = model(feats, lengths, loops=1)
        three, _ = model(feats, lengths, loops=3)
    assert not torch.allclose(one, three, atol=1e-4)


def test_gate_gets_gradient_through_a_loop():
    # Zero-init up weight: its gradient must still be nonzero once loops >= 2,
    # otherwise the gate could never open in training.
    torch.manual_seed(0)
    model = build_model(tiny("looped"))
    feats, lengths, targets, target_lengths = batch()
    loss = model.ctc_loss(*model(feats, lengths, loops=2), targets, target_lengths)
    loss.backward()
    for gate in model.encoder.gates:
        assert gate.up.weight.grad.abs().sum() > 0


ARMS = ["a", "a_best", "b", "b_best", "b_fix", "c", "c_best"]


@pytest.mark.parametrize("arm", ARMS)
def test_arm_configs_hit_param_targets(arm):
    cfg = load_config(CONFIGS / f"arm_{arm}.yaml")
    n = count_params(build_model(cfg.model))
    assert cfg.target_params is not None
    assert abs(n - cfg.target_params) / cfg.target_params < 0.10, n


SWEEP_KEYS = {"lr_peak", "lr_sweep", "lr_sweep_steps_fraction", "lr_from_sweep"}


@pytest.mark.parametrize("arm", ["a", "b", "c"])
def test_best_matches_its_arm_except_lr(arm):
    # DECISIONS.md decision 3: every arm gets the same 3-LR half-step sweep, then a full rerun.
    base = load_config(CONFIGS / f"arm_{arm}.yaml")
    best = load_config(CONFIGS / f"arm_{arm}_best.yaml")
    assert best.model == base.model
    assert best.eval == base.eval
    assert best.train["lr_peak"] is None  # placeholder until the half-step sweep has run
    shared = set(base.train) - SWEEP_KEYS
    assert set(best.train) - SWEEP_KEYS == shared
    assert {k: best.train[k] for k in shared} == {k: base.train[k] for k in shared}
    assert base.train["lr_sweep"] == [3.0e-4, 7.0e-4, 1.5e-3]
    assert base.train["lr_sweep_steps_fraction"] == 0.5


def test_b_fix_differs_from_b_only_in_k():
    b = load_config(CONFIGS / "arm_b.yaml")
    fix = load_config(CONFIGS / "arm_b_fix.yaml")
    assert fix.train["loops_sample"] == [6]
    assert fix.model.default_loops == 6
    assert {**fix.model.__dict__, "default_loops": 10} == b.model.__dict__
    shared = set(b.train) - SWEEP_KEYS - {"loops_sample", "loops_weights"}
    assert {k: fix.train[k] for k in shared} == {k: b.train[k] for k in shared}


def test_b_within_3_percent_of_a_params():
    # DECISIONS.md decision 2: the gates must not break the size match with A.
    a = count_params(build_model(load_config(CONFIGS / "arm_a.yaml").model))
    b = count_params(build_model(load_config(CONFIGS / "arm_b.yaml").model))
    assert a <= b <= 1.03 * a, (a, b)


def test_k_distribution_mean_matches_c_depth():
    # DECISIONS.md decision 4: C is matched to B at B's mean training K.
    b = load_config(CONFIGS / "arm_b.yaml")
    ks, ws = b.train["loops_sample"], b.train["loops_weights"]
    assert ks == [2, 4, 6, 8, 10]
    assert b.model.default_loops == max(ks) == 10  # K_max
    mean_k = sum(k * w for k, w in zip(ks, ws, strict=True)) / sum(ws)
    assert mean_k == 6
    c = load_config(CONFIGS / "arm_c.yaml")
    m = b.model
    assert c.model.n_blocks == m.n_prelude + m.n_core * mean_k + m.n_coda == 14
    # C's exits sit at the depths B runs at k = 2, 4, 6 (decision 5).
    assert c.model.exit_blocks == [m.n_prelude + m.n_core * k + m.n_coda for k in (2, 4, 6)]


def test_flops_match_between_b_and_c_exits():
    b = build_model(load_config(CONFIGS / "arm_b.yaml").model)
    c = build_model(load_config(CONFIGS / "arm_c.yaml").model)
    a = build_model(load_config(CONFIGS / "arm_a.yaml").model)
    assert forward_flops(b, loops=1) == forward_flops(a)  # k=1 is the 4-block path
    for k, depth in [(2, 6), (4, 10), (6, 14)]:
        ratio = forward_flops(b, loops=k) / forward_flops(c, depth=depth)
        assert 1.0 <= ratio < 1.02, (k, ratio)  # gates are the only extra matmuls


def test_plain_early_exit_matches_truncated_forward():
    torch.manual_seed(0)
    cfg = tiny("plain")
    cfg.n_blocks, cfg.exit_blocks = 3, [1, 3]
    model = build_model(cfg).eval()
    feats, lengths, targets, target_lengths = batch()
    with torch.no_grad():
        exits, out_lengths = model.forward_exits(feats, lengths)
        for depth, lp in zip([1, 3], exits, strict=True):
            torch.testing.assert_close(lp, model(feats, lengths, depth=depth)[0])
        torch.testing.assert_close(exits[-1], model(feats, lengths)[0])
    loss = sum(model.ctc_loss(lp, out_lengths, targets, target_lengths) for lp in exits)
    assert torch.isfinite(loss)
    with pytest.raises(ValueError, match="depth"):
        model(feats, lengths, depth=4)


def test_ctc_feasible():
    assert not ctc_feasible(6, [])  # 4x frontend leaves 0 frames
    assert ctc_feasible(7, [1])  # 1 frame, 1 label
    assert not ctc_feasible(7, [1, 2])
    assert not ctc_feasible(11, [3, 3])  # 2 frames: a repeat needs a blank between
    assert ctc_feasible(15, [3, 3])  # 3 frames


def test_device_helper():
    assert get_device("cpu").type == "cpu"
    assert get_device().type in {"cpu", "mps"}
    with pytest.raises(ValueError, match="unknown device"):
        get_device("cuda")


def test_device_helper_refuses_missing_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS was requested"):
        get_device("mps")
    monkeypatch.setenv("LOOP_ASR_DEVICE", "mps")
    with pytest.raises(RuntimeError, match="MPS was requested"):
        get_device()
    monkeypatch.setattr("sys.platform", "darwin")
    monkeypatch.delenv("LOOP_ASR_DEVICE")
    with pytest.warns(UserWarning, match="MPS is not available"):
        assert get_device().type == "cpu"
