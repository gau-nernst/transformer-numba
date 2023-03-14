import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from transformer import EncoderLayer, LayerNorm, Linear, from_pytorch, gelu, self_attention_, softmax_2d_


torch.set_grad_enabled(False)


def assert_allclose(x, y):
    assert np.allclose(x, y, atol=1e-7)  # float32


def test_gelu():
    x = torch.randn(16, 12)
    out_pt = F.gelu(x, approximate="tanh")
    out_np = gelu(x.numpy())
    assert_allclose(out_pt.numpy(), out_np)


def test_softmax():
    x = torch.randn(16, 12)
    out_pt = torch.softmax(x, dim=1)
    out_nb = softmax_2d_(x.numpy())

    assert_allclose(out_pt, out_nb)


def test_layernorm():
    emb_dim = 32
    layer = nn.LayerNorm(emb_dim)
    x = torch.randn((4, emb_dim))

    nn.init.normal_(layer.weight)
    nn.init.normal_(layer.bias)
    out_pt = layer(x)

    layer_nb = from_pytorch(LayerNorm(emb_dim), layer)
    out_nb = layer_nb.forward(x.numpy())

    assert_allclose(out_pt.numpy(), out_nb)


def test_linear():
    in_dim, out_dim = 16, 32
    layer = nn.Linear(in_dim, out_dim)
    x = torch.randn((4, in_dim))
    out_pt = layer(x)

    layer_nb = from_pytorch(Linear(in_dim, out_dim), layer)
    out_nb = layer_nb.forward(x.numpy())

    assert_allclose(out_pt.numpy(), out_nb)


def test_self_attention():
    emb_dim, n_heads = 32, 4
    layer = nn.MultiheadAttention(emb_dim, n_heads, bias=False)
    x = torch.randn(4, emb_dim)
    out_pt, _ = layer(x, x, x, need_weights=False)

    x_np = x.numpy()
    w_qkv = layer.in_proj_weight.numpy()
    w_out = layer.out_proj.weight.numpy()
    out_nb = self_attention_(x_np, w_qkv, w_out, layer.num_heads)

    assert_allclose(out_pt.numpy(), out_nb)


def test_encoder_layer():
    emb_dim, n_heads = 32, 4
    layer = nn.TransformerEncoderLayer(
        emb_dim,
        n_heads,
        dim_feedforward=emb_dim * 4,
        activation=nn.GELU(approximate="tanh"),
        norm_first=True,
    ).eval()
    x = torch.randn(4, emb_dim)
    out_pt = layer(x)

    x_np = x.numpy()
    layer_nb = from_pytorch(EncoderLayer(emb_dim, n_heads), layer)
    out_nb = layer_nb.forward(x_np)

    assert_allclose(out_pt.numpy(), out_nb)
