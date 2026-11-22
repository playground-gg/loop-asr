"""CPU smoke tests: forward, CTC loss, one optimizer step, loop count, closed gate at init."""

from pathlib import Path

import pytest
import torch

from loop_asr import ModelConfig, build_model, count_params, get_device, load_config

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
    torch.nn.init.normal_(model.encoder.gate.up.weight, std=0.5)
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
    assert model.encoder.gate.up.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("arm", ["a", "a_best", "b", "c"])
def test_arm_configs_hit_param_targets(arm):
    cfg = load_config(CONFIGS / f"arm_{arm}.yaml")
    n = count_params(build_model(cfg.model))
    assert cfg.target_params is not None
    assert abs(n - cfg.target_params) / cfg.target_params < 0.10, n


def test_a_best_matches_a_except_lr():
    a = load_config(CONFIGS / "arm_a.yaml")
    best = load_config(CONFIGS / "arm_a_best.yaml")
    assert best.model == a.model
    assert best.eval == a.eval
    assert best.train["lr_peak"] is None  # placeholder until the half-step sweep has run
    shared = set(a.train) - {"lr_peak", "lr_sweep", "lr_sweep_steps_fraction"}
    assert {k: best.train[k] for k in shared} == {k: a.train[k] for k in shared}
    assert a.train["lr_sweep_steps_fraction"] == 0.5


def test_arm_b_loop_schedule():
    b = load_config(CONFIGS / "arm_b.yaml")
    assert b.train["loops_sample"] == [2, 4, 6, 8, 10]
    assert b.model.default_loops == max(b.train["loops_sample"]) == 10  # K_max
    c = load_config(CONFIGS / "arm_c.yaml")
    m = b.model
    assert c.model.n_blocks == m.n_prelude + m.n_core * 8 + m.n_coda  # C matches B at K=8


def test_device_helper():
    assert get_device("cpu").type == "cpu"
    assert get_device().type in {"cpu", "mps"}
