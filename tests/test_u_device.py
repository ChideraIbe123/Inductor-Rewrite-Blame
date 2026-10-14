"""Device selection is an environment variable; on a CPU-only machine everything must still default to cpu."""
import os

import torch

from rewrite_blame import env
from rewrite_blame.models import to_device


def test_default_device_is_cpu(monkeypatch):
    monkeypatch.delenv("REWRITE_BLAME_DEVICE", raising=False)
    assert env.device_name() == "cpu" and not env.is_cuda()
    assert "cuda" not in env.machine_id()


def test_cuda_device_changes_fingerprint(monkeypatch):
    monkeypatch.setenv("REWRITE_BLAME_DEVICE", "cuda:0")
    assert env.is_cuda()
    assert env.machine_id().endswith("cuda")          # GPU name + "cuda", not the host CPU
    assert env.fingerprint(4).startswith(env.machine_id())


def test_to_device_moves_nested_tensors():
    x = torch.zeros(2)
    out = to_device((x, [x, {"a": x, "n": 3}], "s"), torch.device("cpu"))
    assert isinstance(out, tuple) and isinstance(out[1], list) and out[1][1]["n"] == 3 and out[2] == "s"
    assert out[1][1]["a"].device.type == "cpu"


def test_build_stays_on_cpu_without_cuda(monkeypatch):
    monkeypatch.delenv("REWRITE_BLAME_DEVICE", raising=False)
    from rewrite_blame import models
    m, inputs = models.build("attention_block")
    assert all(t.device.type == "cpu" for t in inputs if isinstance(t, torch.Tensor))
    assert next(m.parameters()).device.type == "cpu"


def test_conv_sweep_batch_depends_on_device(monkeypatch):
    from rewrite_blame.models.sweeps import conv_batch, config_of
    monkeypatch.delenv("REWRITE_BLAME_SWEEP_BATCH", raising=False)
    monkeypatch.delenv("REWRITE_BLAME_DEVICE", raising=False)
    assert conv_batch(256, 56) == 4 and config_of("sweep_conv_C256_R56_K3_L3_dense")["act_numel"] == 4 * 256 * 56 * 56
    monkeypatch.setenv("REWRITE_BLAME_DEVICE", "cuda")
    assert conv_batch(256, 56) == 64                      # 64*256*56*56 = 51M elements already
    assert conv_batch(16, 14) == 4096                     # tiny configuration: capped at 4096
    assert 64 <= conv_batch(64, 28) <= 4096 and conv_batch(64, 28) * 64 * 28 * 28 >= 32 * 1024 * 1024
    monkeypatch.setenv("REWRITE_BLAME_SWEEP_BATCH", "8")
    assert conv_batch(256, 56) == 8
