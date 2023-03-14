import numba as nb
import numpy as np


FLOAT = nb.float32
FLOAT_1D = FLOAT[::1]
FLOAT_2D = FLOAT[:, ::1]


@nb.vectorize([FLOAT(FLOAT)])
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)))


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
def layernorm_(x, w, b, eps):
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


@nb.experimental.jitclass
class LayerNorm:
    w: FLOAT_1D
    b: FLOAT_1D
    eps: FLOAT

    def __init__(self, emb_dim):
        self.w = np.empty(emb_dim, dtype=FLOAT)
        self.b = np.empty(emb_dim, dtype=FLOAT)
        self.eps = 1e-5

    def forward(self, x):
        for i in range(x.shape[0]):
            layernorm_(x[i], self.w, self.b, self.eps)
        return x


@nb.njit(FLOAT_2D(FLOAT_2D, FLOAT_2D, FLOAT_2D, nb.int64), fastmath=True)
def self_attention_(x, w_qkv, w_out, n_heads):
    # x: (seq_len, emb_dim)
    # w_qkv: (emb_dim x 3, emb_dim)
    # w_out: (emb_dim, emb_dim)
    seq_len, emb_dim = x.shape
    head_dim = emb_dim // n_heads

    qkv = w_qkv @ x.T  # (emb_dim x 3, seq_len)
    q = qkv[:emb_dim]  # (emb_dim, seq_len) each
    k = qkv[emb_dim : emb_dim * 2]
    v = qkv[emb_dim * 2 :]
    k /= np.sqrt(head_dim)

    out = np.empty((emb_dim, seq_len), dtype=FLOAT)
    for i in range(n_heads):
        s = slice(head_dim * i, head_dim * (i + 1))
        attns = softmax_2d_(q[s].T @ k[s])  # (seq_len, seq_len)
        np.dot(v[s], attns.T, out[s])  # (head_dim, seq_len)

    return out.T @ w_out.T  # (seq_len, emb_dim)


@nb.experimental.jitclass
class Linear:
    w: FLOAT_2D
    b: FLOAT_1D

    def __init__(self, in_dim, out_dim):
        self.w = np.empty((out_dim, in_dim), dtype=FLOAT)
        self.b = np.empty(out_dim, dtype=FLOAT)

    def forward(self, x):
        out = x @ self.w.T
        np.add(out, self.b.reshape(1, -1), out)
        return out


@nb.experimental.jitclass
class EncoderLayer:
    w_qkv: FLOAT_2D
    w_out: FLOAT_2D
    n_heads: nb.int64
    linear1: Linear
    linear2: Linear
    norm1: LayerNorm
    norm2: LayerNorm

    def __init__(self, emb_dim, n_heads):
        self.w_qkv = np.empty((emb_dim * 3, emb_dim), dtype=FLOAT)
        self.w_out = np.empty((emb_dim, emb_dim), dtype=FLOAT)
        self.n_heads = n_heads
        self.linear1 = Linear(emb_dim, emb_dim * 4)
        self.linear2 = Linear(emb_dim * 4, emb_dim)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)

    def forward(self, x):
        x = x + self_attention_(self.norm1.forward(x.copy()), self.w_qkv, self.w_out, self.n_heads)
        x = x + self.linear2.forward(gelu(self.linear1.forward(self.norm2.forward(x.copy()))))
        return x


def from_pytorch(nb_module, pt_module):
    if isinstance(nb_module, LayerNorm):
        nb_module.w[:] = pt_module.weight.detach().numpy()
        nb_module.b[:] = pt_module.bias.detach().numpy()
        nb_module.eps = pt_module.eps

    elif isinstance(nb_module, Linear):
        nb_module.w[:] = pt_module.weight.detach().numpy()
        nb_module.b[:] = pt_module.bias.detach().numpy()

    elif isinstance(nb_module, EncoderLayer):
        nb_module.w_qkv[:] = pt_module.self_attn.in_proj_weight.detach().numpy()
        nb_module.w_out[:] = pt_module.self_attn.out_proj.weight.detach().numpy()
        nb_module.n_heads = pt_module.self_attn.num_heads
        from_pytorch(nb_module.linear1, pt_module.linear1)
        from_pytorch(nb_module.linear2, pt_module.linear2)
        from_pytorch(nb_module.norm1, pt_module.norm1)
        from_pytorch(nb_module.norm2, pt_module.norm2)

    return nb_module
