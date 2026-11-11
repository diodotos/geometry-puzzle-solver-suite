"""The cost-to-go network.

DeepCubeA [Ago19] uses a fully connected network: one-hot state -> 5000 -> 1000 -> four
residual blocks of width 1000 -> 1, with batch norm and ReLU. ``CostToGo`` has the same
shape with configurable widths; the 2x2x2 needs far less capacity than the 3x3x3.

The input is the categorical encoding produced by Rust (``int64[batch, len]``, values in
``0..classes``). The first layer is a one-hot linear layer implemented as an embedding
lookup summed over positions, which is mathematically identical to
``Linear(one_hot(x).flatten())`` but avoids materialising the one-hot tensor. Training uses
``embedding_bag`` (no ``[batch, len, hidden]`` intermediate, which is 4 GB for a 3x3 batch);
ONNX export uses the equivalent Gather + ReduceSum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Architecture:
    len: int
    classes: int
    hidden1: int = 512
    hidden2: int = 256
    blocks: int = 3

    def to_dict(self) -> dict:
        return {"type": "resmlp-onehot", **asdict(self)}


class OneHotLinear(nn.Module):
    def __init__(self, length: int, classes: int, out: int):
        super().__init__()
        self.embed = nn.Embedding(length * classes, out)
        self.bias = nn.Parameter(torch.zeros(out))
        self.register_buffer("offsets", torch.arange(length, dtype=torch.long) * classes, persistent=False)
        nn.init.normal_(self.embed.weight, std=(1.0 / length) ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        idx = x.long() + self.offsets
        if torch.onnx.is_in_onnx_export():
            return self.embed(idx).sum(dim=1) + self.bias
        return F.embedding_bag(idx, self.embed.weight, mode="sum") + self.bias


class ResidualBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.bn1 = nn.BatchNorm1d(width)
        self.fc2 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.relu(self.bn1(self.fc1(x)))
        y = self.bn2(self.fc2(y))
        return torch.relu(x + y)


class CostToGo(nn.Module):
    def __init__(self, arch: Architecture):
        super().__init__()
        self.arch = arch
        self.input = OneHotLinear(arch.len, arch.classes, arch.hidden1)
        self.bn_in = nn.BatchNorm1d(arch.hidden1)
        self.fc = nn.Linear(arch.hidden1, arch.hidden2)
        self.bn = nn.BatchNorm1d(arch.hidden2)
        self.blocks = nn.Sequential(*[ResidualBlock(arch.hidden2) for _ in range(arch.blocks)])
        self.out = nn.Linear(arch.hidden2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn_in(self.input(x)))
        h = torch.relu(self.bn(self.fc(h)))
        h = self.blocks(h)
        return self.out(h).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
