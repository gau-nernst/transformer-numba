import numpy as np
import torch
from torch import nn

from transformer import layernorm_2d_, self_attention, softmax_2d_


def assert_allclose(x, y):
    assert np.allclose(x, y, atol=1e-7)  # float32


@torch.inference_mode()
def test_softmax():
    x = torch.randn(16, 12)
    out_pt = torch.softmax(x, dim=1)
    out_nb = softmax_2d_(x.numpy())

    assert_allclose(out_pt, out_nb)


@torch.inference_mode()
def test_layernorm():
    emb_dim = 32
    layer = nn.LayerNorm(emb_dim)
    x = torch.randn((4, emb_dim))

    nn.init.normal_(layer.weight)
    nn.init.normal_(layer.bias)
    out = layer(x)

    x_np = x.numpy()
    w = layer.weight.numpy()
    b = layer.bias.numpy()
    out_nb = layernorm_2d_(x_np, w, b, layer.eps)

    assert_allclose(out.numpy(), out_nb)


@torch.inference_mode()
def test_self_attention():
    emb_dim, n_heads = 32, 4
    layer = nn.MultiheadAttention(emb_dim, n_heads, bias=False)
    x = torch.randn(4, emb_dim)
    out, _ = layer(x, x, x, need_weights=False)

    x_np = x.numpy()
    w_qkv = layer.in_proj_weight.numpy()
    w_out = layer.out_proj.weight.numpy()

    out_nb = self_attention(x_np, w_qkv, w_out, layer.num_heads)

    assert_allclose(out.numpy(), out_nb)
