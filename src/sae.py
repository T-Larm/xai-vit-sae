import torch
import torch.nn as nn


class SAE(nn.Module):
    def __init__(self, d_input: int, d_hidden: int, alpha: float = 1e-3):
        super().__init__()
        self.alpha = alpha
        self.W_e = nn.Parameter(torch.randn(d_hidden, d_input) * 0.02)
        self.b_e = nn.Parameter(torch.zeros(d_hidden))
        self.W_d = nn.Parameter(torch.randn(d_input, d_hidden) * 0.02)
        self.b_d = nn.Parameter(torch.zeros(d_input))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, d_input) → f: (batch, d_hidden)
        return torch.relu((x - self.b_d) @ self.W_e.T + self.b_e)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        # f: (batch, d_hidden) → x_hat: (batch, d_input)
        return f @ self.W_d.T + self.b_d

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        return f, self.decode(f)

    def loss(self, x: torch.Tensor, f: torch.Tensor, x_hat: torch.Tensor):
        mse = ((x - x_hat) ** 2).mean()
        l1 = f.abs().sum(dim=-1).mean()  # per-token sum, then mean over batch
        return mse + self.alpha * l1, mse, l1

    def normalize_decoder(self):
        with torch.no_grad():
            norms = self.W_d.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.W_d.div_(norms)
