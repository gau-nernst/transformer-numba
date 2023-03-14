import numba as nb
import numpy as np


FLOAT_1D = nb.float32[::1]
FLOAT_2D = nb.float32[:, ::1]


@nb.njit(FLOAT_1D(FLOAT_1D), fastmath=True)
def softmax_1d_(x):
    x -= x.max()
    accum = 0.0
    for i in range(x.shape[0]):
        x[i] = np.exp(x[i])
        accum += x[i]
    x /= accum
    return x


@nb.njit(FLOAT_2D(FLOAT_2D), fastmath=True)
def softmax_2d_(x):
    for i in range(x.shape[0]):
        softmax_1d_(x[i])
    return x


@nb.njit(FLOAT_1D(FLOAT_1D, FLOAT_1D, FLOAT_1D, nb.float32), fastmath=True)
def layernorm_1d_(x, w, b, eps):
    (M,) = x.shape
    sum_x = 0.0
    sum_x2 = 0.0
    for _x in x:
        sum_x += _x
        sum_x2 += _x * _x

    mean = sum_x / M
    var = sum_x2 / M - mean * mean
    scale = 1.0 / np.sqrt(var + eps)
    for i in range(M):
        x[i] = (x[i] - mean) * scale * w[i] + b[i]
    return x


@nb.njit(FLOAT_2D(FLOAT_2D, FLOAT_1D, FLOAT_1D, nb.float32), fastmath=True)
def layernorm_2d_(x, w, b, eps):
    for i in range(x.shape[0]):
        layernorm_1d_(x[i], w, b, eps)
    return x


@nb.njit(FLOAT_2D(FLOAT_2D, FLOAT_2D, FLOAT_2D, nb.int64), fastmath=True)
def self_attention(x, w_qkv, w_out, n_heads):
    # x: (seq_len, emb_dim)
    # w_qkv: (emb_dim x 3, emb_dim)
    # w_out: (emb_dim, emb_dim)
    seq_len, emb_dim = x.shape
    head_dim = emb_dim // n_heads

    qkv = w_qkv @ x.T  # (emb_dim x 3, seq_len)
    q = qkv[:emb_dim]  # (emb_dim, seq_len) each
    k = qkv[emb_dim : emb_dim * 2]
    v = qkv[emb_dim * 2 :]
    k *= 1.0 / np.sqrt(head_dim)

    out = np.empty((emb_dim, seq_len), dtype=np.float32)
    for i in range(n_heads):
        s = slice(head_dim * i, head_dim * (i + 1))
        attns = softmax_2d_(q[s].T @ k[s])  # (seq_len, seq_len)
        np.dot(v[s], attns.T, out[s])  # (head_dim, seq_len)

    return out.T @ w_out.T  # (seq_len, emb_dim)
