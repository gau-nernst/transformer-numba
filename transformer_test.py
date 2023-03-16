import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from transformer import (
    LayerNormParams,
    LinearParams,
    SelfAttentionParams,
    TransformerBlockParams,
    from_pytorch,
    gelu,
    layernorm2d_,
    linear,
    self_attention_,
    softmax2d_,
    transformer_block,
)


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
    out_nb = softmax2d_(x.numpy())

    assert_allclose(out_pt, out_nb)


def test_layernorm():
    emb_dim = 32
    layer = nn.LayerNorm(emb_dim)
    x = torch.randn((4, emb_dim))

    nn.init.normal_(layer.weight)
    nn.init.normal_(layer.bias)
    out_pt = layer(x)

    params = from_pytorch(LayerNormParams(emb_dim), layer)
    out_nb = layernorm2d_(x.numpy(), params)

    assert_allclose(out_pt.numpy(), out_nb)


def test_linear():
    in_dim, out_dim = 16, 32
    layer = nn.Linear(in_dim, out_dim)
    x = torch.randn((4, in_dim))
    out_pt = layer(x)

    params = from_pytorch(LinearParams(in_dim, out_dim), layer)
    out_nb = linear(x.numpy(), params)

    assert_allclose(out_pt.numpy(), out_nb)


def test_self_attention():
    emb_dim, n_heads = 32, 4
    layer = nn.MultiheadAttention(emb_dim, n_heads, bias=False)
    x = torch.randn(4, emb_dim)
    out_pt, _ = layer(x, x, x, need_weights=False)

    params = from_pytorch(SelfAttentionParams(emb_dim, n_heads), layer)
    out_nb = self_attention_(x.numpy(), params)

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

    params = from_pytorch(TransformerBlockParams(emb_dim, n_heads), layer)
    out_nb = transformer_block(x.numpy(), params)

    assert_allclose(out_pt.numpy(), out_nb)
