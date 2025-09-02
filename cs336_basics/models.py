import torch
from einops import einsum
import math

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        weights_init = torch.nn.init.trunc_normal_(
            torch.randn(out_features, in_features, device=device,dtype=dtype),
            mean=0,
            std=math.sqrt(2./(in_features + out_features)),
            a=-3*math.sqrt(2./(in_features + out_features)),
            b=3*math.sqrt(2./(in_features + out_features))
        )
        self.weights = torch.nn.Parameter(weights_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = einsum(x, self.weights, "batch seq d_in, d_out d_in -> batch seq d_out")
        return output

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        weights_init = torch.nn.init.trunc_normal_(
            torch.randn(num_embeddings, embedding_dim, device=device,dtype=dtype),
            mean=0,
            std=1,
            a=-3,
            b=3
        )
        self.weights = torch.nn.Parameter(weights_init)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]

class RMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weights = torch.nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_squared = x**2 + self.eps
        x_mean = torch.mean(x_squared, dim=-1, keepdim=True)
        x_rms = torch.sqrt(x_mean)
        x /= x_rms
        # result = einsum(x, self.weights, "batch seq d_model, d_model -> batch seq d_model")
        result = x * self.weights
        return result.to(in_dtype)

class SwiGLU(torch.nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(in_features=d_ff, out_features=d_model, device=None, dtype=None)
        self.w3 = Linear(in_features=d_ff, out_features=d_model, device=None, dtype=None)
        self.w2 = Linear(in_features=d_model, out_features=d_ff, device=None, dtype=None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # non-linearity
        a = self.w1(x)
        silu = a * torch.sigmoid(a)
        # linear output
        b = self.w3(x)
        # point-wise multiplication
        c = silu * b
        # final output
        output = self.w2(c)
        return output



