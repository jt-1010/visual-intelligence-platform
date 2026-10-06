"""Sign classifier with Mamba in place of the Transformer; same input and output as ml/asl/model.py.

Pure PyTorch (mamba-ssm does not build on Windows). Bidirectional by default.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the Transformer model's stem so the comparison changes one thing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "asl"))
from model import ConvStem  # noqa: E402


class RMSNorm(nn.Module):
    """Rescales each frame's features to a steady size, so one layer's output
    can't be much louder than the next layer expects."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaMixer(nn.Module):
    """One Mamba layer. Reads the frames in order and keeps a running memory of
    what it has seen. Each frame's output depends only on that frame and earlier
    ones, never later ones. Shape: (batch, frames, features) in and out."""

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.d_inner = expand * dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = math.ceil(dim / 16)

        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)

        # Looks at the last few frames (d_conv=4) to pick up short, quick motion
        # before the memory step. Padding is what lets it look backwards without
        # shrinking the sequence; the extra frames the padding adds at the end are
        # cut off in forward(), so a frame never gets to peek at later frames.
        self.conv = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=True,
        )

        # These three numbers are worked out from the current frame itself:
        #   dt: how much of the old memory to replace with this frame
        #   B:  what from this frame to write into memory
        #   C:  what to read back out of memory
        # That is what makes it "selective": it can hold on to important frames
        # and mostly skip frames where nothing is happening, instead of treating
        # every frame the same.
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A controls how fast the memory fades between frames. The 16 memory slots
        # start at different rates (1 to 16), so some forget quickly (recent
        # motion) and some hold on longer (earlier in the sign). It is stored as
        # a log and negated when used (A = -exp(A_log)) so it can never turn
        # positive. A positive value would make the memory grow every frame
        # instead of fading, and training would blow up.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        # D lets the current frame skip the memory and go straight to the output,
        # so the layer can fall back on "just use this frame" when memory doesn't help.
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        self._init_dt()

    def _init_dt(self, dt_min: float = 1e-3, dt_max: float = 1e-1) -> None:
        """Set a starting value for dt, the size of the step the memory takes each frame.

        A small dt means the layer holds on to the past for a long time; a large
        dt means the memory is mostly replaced by the newest frame. Each channel
        starts at a random value between dt_min and dt_max, spread evenly on a log
        scale, so the layer begins with a mix of long and short memories. If they
        all started at the same value, training would have to spend its early
        epochs just finding a useful time scale.
        """
        std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=1e-4)
        # forward() squashes dt through softplus to keep it positive. Store the
        # inverse here so that after the squash the value is the one we picked.
        inv_softplus = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_softplus)

    def selective_scan(
        self, x: torch.Tensor, dt: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> torch.Tensor:
        """Walk through the frames one at a time. At each frame the memory (h) fades
        a little, the current frame is written into it, and the output is read
        back out. A plain Python loop is fine because there are only 32 frames."""
        bsz, T, d = x.shape
        A = -torch.exp(self.A_log)                       # (d, n)
        h = x.new_zeros(bsz, d, self.d_state)
        ys = []
        for t in range(T):
            dt_t = dt[:, t]                              # (B, d)
            dA = torch.exp(dt_t.unsqueeze(-1) * A)       # (B, d, n)
            dBx = dt_t.unsqueeze(-1) * B[:, t].unsqueeze(1) * x[:, t].unsqueeze(-1)
            h = dA * h + dBx
            ys.append((h * C[:, t].unsqueeze(1)).sum(-1))  # (B, d)
        y = torch.stack(ys, dim=1)
        return y + x * self.D

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        T = u.shape[1]
        # Split the input in two: x goes through the memory, and z is a gate that
        # decides how much of the memory's result is let through at the end.
        x, z = self.in_proj(u).chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))[..., :T].transpose(1, 2)
        x = F.silu(x)

        dt, B, C = torch.split(
            self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt))

        y = self.selective_scan(x, dt, B, C)
        return self.out_proj(y * F.silu(z))


class MambaBlock(nn.Module):
    """Rescale, run the Mamba layer, then add the result back onto the input. Adding
    it back lets the original signal pass straight through, which keeps training
    stable. If bidirectional, the window is also read backwards and the two
    results are added, so each frame gets context from before and after it."""

    def __init__(self, dim: int, bidirectional: bool, dropout: float, **mixer_kwargs) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        self.fwd = MambaMixer(dim, **mixer_kwargs)
        self.bwd = MambaMixer(dim, **mixer_kwargs) if bidirectional else None
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        y = self.fwd(h)
        if self.bwd is not None:
            y = y + self.bwd(h.flip(1)).flip(1)
        return x + self.drop(y)


class SignMambaClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_dim: int = 153,
        frames: int = 32,  # unused by the maths; kept so checkpoints carry it
        dim: int = 192,
        depth: int = 3,
        d_state: int = 16,
        expand: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.stem = ConvStem(in_dim, dim)
        self.blocks = nn.ModuleList(
            MambaBlock(dim, bidirectional, dropout, d_state=d_state, expand=expand)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        # Named `head` to match ml/asl/model.py, so the pretrained-weight loader in
        # ml/asl/train.py (which skips anything starting with head.) also works here.
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # No frame-number markers are added (the Transformer needs them). These
        # layers read frames in order, so they already know which came first.
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        # Average over all frames: a sign is the whole motion, not one key frame.
        x = self.norm(x.mean(dim=1))
        return self.head(self.drop(x))


def build(num_classes: int, **kwargs) -> SignMambaClassifier:
    model = SignMambaClassifier(num_classes=num_classes, **kwargs)
    params = sum(p.numel() for p in model.parameters())
    print(f"SignMambaClassifier: {num_classes} classes, {params / 1e6:.2f}M parameters")
    return model
