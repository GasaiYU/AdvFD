"""Check GAN gradient isolation without downloading pretrained weights."""

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn


@pytest.mark.parametrize("checkpointing", [False, True])
def test_inception_gan_updates_features_but_keeps_generator_step_isolated(monkeypatch, checkpointing):
    path = Path(__file__).resolve().parents[1] / "models" / "patch_gan.py"
    spec = importlib.util.spec_from_file_location("patch_gan_under_test", path)
    gan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gan)

    class ToyInception(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 2048, 1)
            self.bn = nn.BatchNorm2d(2048)
            self.fc = nn.Linear(2048, 1008)

        def forward_features_by_layer(self, images, layers):
            features = self.bn(self.conv(images)).relu().mean(dim=(2, 3))
            return {"Mixed_7c": (features, None)}

    def load_inception(**kwargs):
        assert kwargs["normalize"] is False
        assert kwargs["pretrained"] is True
        return ToyInception().eval().requires_grad_(False)

    perception = ModuleType("utils.perception_util")
    perception.load_inception = load_inception
    monkeypatch.setitem(sys.modules, "utils.perception_util", perception)

    discriminator = gan.InceptionDiscriminator(device="cpu").train()
    assert discriminator.backbone.bn.training
    running_mean = discriminator.backbone.bn.running_mean.clone()
    backbone_before = discriminator.backbone.conv.weight.detach().clone()
    head_before = discriminator.head.weight.detach().clone()
    optimizer = torch.optim.AdamW(discriminator.parameters(), lr=1e-3)
    fake = torch.randn(2, 3, 4, 4, requires_grad=True)
    real = torch.randn_like(fake)

    d_loss, _, _ = gan.patch_discriminator_loss(discriminator, real, fake)
    d_loss.backward()
    assert fake.grad is None
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    assert not torch.equal(backbone_before, discriminator.backbone.conv.weight)
    assert not torch.equal(head_before, discriminator.head.weight)

    discriminator.requires_grad_(False)
    g_loss = gan.patch_generator_loss(discriminator, fake, grad_checkpointing=checkpointing)
    bn_updates = discriminator.backbone.bn.num_batches_tracked.clone()
    g_loss.backward()
    assert fake.grad is not None and fake.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in discriminator.parameters())
    assert not torch.equal(running_mean, discriminator.backbone.bn.running_mean)
    assert bn_updates.item() == 3  # Real D forward, fake D forward, fake G forward.
    assert torch.equal(bn_updates, discriminator.backbone.bn.num_batches_tracked)


@pytest.mark.parametrize("step", [1000, 1001, 5000])
def test_training_gan_blocks_match_patchgan_with_adaptive_weight_and_checkpointing(step):
    """Execute the actual GAN blocks on CPU without importing the CUDA trainer."""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("gan_training_test", root / "models/patch_gan.py")
    gan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gan)
    tree = ast.parse((root / "main_fd.py").read_text())
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_fd_train_step")
    train_step = next(n for n in factory.body if isinstance(n, ast.FunctionDef) and n.name == "fd_train_step")
    gan_blocks = [n for n in train_step.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "gan_active"]
    backward = next(n for n in train_step.body if isinstance(n, ast.If) and any(
        isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name) and call.func.value.id == "loss"
        and call.func.attr == "backward" for call in ast.walk(n)
    ))
    backward_index = train_step.body.index(backward)
    gan_blocks = [n for n in gan_blocks if train_step.body.index(n) < backward_index]
    restore_blocks = [n for n in train_step.body[backward_index + 1:] if isinstance(n, ast.If)
                      and isinstance(n.test, ast.Name) and n.test.id == "gan_active"]
    discriminator = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 1))
    optimizer = torch.optim.AdamW(discriminator.parameters(), lr=1e-3)
    before = discriminator[0].weight.detach().clone()
    fake = torch.linspace(-0.5, 1.5, 12).reshape(4, 3).requires_grad_(True)
    main_loss = fake.square().mean()
    args = SimpleNamespace(
        current_step=step, patch_gan_start_step=1000, patch_gan_backbone="inception",
        patch_gan_weight=0.1,
        patch_gan_warmup_steps=9000, patch_gan_adaptive_weight=True,
        patch_gan_max_adaptive_weight=1e4,
        patch_gan_grad_checkpointing=True, fd_sequential_backward=False,
    )
    namespace = dict(
        torch=torch, args=args, gan_active=True, sampled=fake,
        real_images_for_gan=None, real_batch_fn=lambda: (torch.randn_like(fake), None),
        patch_discriminator=discriminator, patch_d_optimizer=optimizer,
        patch_discriminator_loss=gan.patch_discriminator_loss,
        patch_generator_loss=gan.patch_generator_loss,
        _set_requires_grad=lambda module, enabled: module.requires_grad_(enabled),
        _all_reduce_grads=lambda module: None,
        loss_dict={}, loss=main_loss, fd_main_loss=main_loss,
        model_wo_ddp=object(), _get_generator_last_layer=lambda _: fake,
        d_loss=None, d_real=None, d_fake=None,
    )
    adaptive = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_adaptive_adversarial_weight")
    exec(compile(ast.Module(body=[adaptive], type_ignores=[]), "adaptive_weight", "exec"), namespace)
    program = ast.Module(body=[*gan_blocks, backward, *restore_blocks], type_ignores=[])
    exec(compile(program, "main_fd_gan_blocks", "exec"), namespace)
    assert "patch_d_loss" in namespace["loss_dict"]
    assert not torch.equal(before, discriminator[0].weight)
    assert torch.equal(namespace["sampled_for_gan"], fake.clamp(0, 1))
    assert namespace["loss_dict"]["patch_g_warmup"] == pytest.approx((step - 1000 + 1) / 9000)
    assert fake.grad is not None
    if step > 1000:
        assert fake.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in discriminator.parameters())


def test_rebuttal_launcher_matches_original_patchgan_training_arguments(tmp_path):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "main_fd.py").read_text())
    parser_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_args_parser")
    namespace = {"argparse": argparse}
    exec(compile(ast.Module(body=[parser_node], type_ignores=[]), "parser", "exec"), namespace)
    parser = namespace["get_args_parser"]()
    torchrun = tmp_path / "torchrun"
    torchrun.write_text(f"#!{sys.executable}\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n")
    torchrun.chmod(0o755)
    env = dict(os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
               DRY_RUN="1", MODEL_SIZE="B", PATCH_GAN_INIT="base", SEED="1",
               ENABLE_WANDB="1", NNODES="1", GPUS_PER_NODE="8", GLOBAL_BSZ="1024",
               DATA_ROOT="/mmu-vcg/zhangxu34/datasets/ImageNet-1K/")
    old = subprocess.run(["bash", "scripts/ablation/table_3_JiT_patch_gan.sh"], cwd=root,
                         env=env, check=True, capture_output=True, text=True)
    old_command = json.loads(old.stdout)
    new = subprocess.run(["bash", "scripts/rebuttal/w1_jit_b_sim_inception_gan.sh"], cwd=root,
                         env=env, check=True, capture_output=True, text=True)
    new_command = shlex.split(new.stdout.split("[W1] Command: ", 1)[1])
    old_args = vars(parser.parse_args(old_command[old_command.index("main_fd.py") + 1:]))
    new_args = vars(parser.parse_args(new_command[new_command.index("main_fd.py") + 1:]))
    for key in ("project", "exp_name", "output_dir", "patch_gan_backbone"):
        old_args.pop(key)
        new_args.pop(key)
    assert new_args == old_args
