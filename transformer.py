import numba as nb
import numpy as np


FLOAT = nb.float32
FLOAT_1D = FLOAT[::1]
FLOAT_2D = FLOAT[:, ::1]
LONG = nb.int64
LONG_1D = LONG[::1]


@nb.vectorize([FLOAT(FLOAT)], fastmath=True)
def gelu(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)))


@nb.njit(FLOAT_1D(FLOAT_1D), fastmath=True)
def softmax1d_(x):
    x_max = x.max()
    accum = 0.0
    for i in range(x.shape[0]):
        x[i] = np.exp(x[i] - x_max)
        accum += x[i]
    x /= accum
    return x


@nb.njit(FLOAT_2D(FLOAT_2D), fastmath=True)
def softmax2d_(x):
    for i in range(x.shape[0]):
        softmax1d_(x[i])
    return x


@nb.experimental.jitclass
class LayerNormParams:
    w: FLOAT_1D
    b: FLOAT_1D
    eps: FLOAT

    def __init__(self, emb_dim):
        self.w = np.empty(emb_dim, dtype=FLOAT)
        self.b = np.empty(emb_dim, dtype=FLOAT)
        self.eps = 1e-5


_LayerNormParams = nb.typeof(LayerNormParams(4))


@nb.njit(FLOAT_1D(FLOAT_1D, _LayerNormParams), fastmath=True)
def layernorm1d_(x, params):
    (M,) = x.shape
    sum_x = 0.0
    sum_x2 = 0.0
    for _x in x:
        sum_x += _x
        sum_x2 += _x * _x

    mean = sum_x / M
    var = sum_x2 / M - mean * mean
    scale = 1.0 / np.sqrt(var + params.eps)
    for i in range(M):
        x[i] = (x[i] - mean) * scale * params.w[i] + params.b[i]
    return x


@nb.njit(FLOAT_2D(FLOAT_2D, _LayerNormParams), fastmath=True)
def layernorm2d_(x, params):
    for i in range(x.shape[0]):
        layernorm1d_(x[i], params)
    return x


@nb.experimental.jitclass
class SelfAttentionParams:
    w_qkv: FLOAT_2D
    w_out: FLOAT_2D
    n_heads: LONG

    def __init__(self, emb_dim, n_heads):
        self.w_qkv = np.empty((emb_dim * 3, emb_dim), dtype=FLOAT)
        self.w_out = np.empty((emb_dim, emb_dim), dtype=FLOAT)
        self.n_heads = n_heads


_SelfAttentionParams = nb.typeof(SelfAttentionParams(4, 2))


@nb.njit(FLOAT_2D(FLOAT_2D, _SelfAttentionParams), fastmath=True)
def self_attention_(x, params):
    # x: (seq_len, emb_dim)
    seq_len, emb_dim = x.shape
    head_dim = emb_dim // params.n_heads

    qkv = params.w_qkv @ x.T  # (emb_dim x 3, seq_len)
    q = qkv[:emb_dim]  # (emb_dim, seq_len) each
    k = qkv[emb_dim : emb_dim * 2]
    v = qkv[emb_dim * 2 :]
    k /= np.sqrt(head_dim)

    out = np.empty((emb_dim, seq_len), dtype=FLOAT)
    for i in range(params.n_heads):
        s = slice(head_dim * i, head_dim * (i + 1))
        attns = softmax2d_(q[s].T @ k[s])  # (seq_len, seq_len)
        np.dot(v[s], attns.T, out[s])  # (head_dim, seq_len)

    return np.dot(out.T, params.w_out.T, x)  # (seq_len, emb_dim)


@nb.experimental.jitclass
class LinearParams:
    w: FLOAT_2D
    b: FLOAT_1D

    def __init__(self, in_dim, out_dim):
        self.w = np.empty((out_dim, in_dim), dtype=FLOAT)
        self.b = np.empty(out_dim, dtype=FLOAT)


_LinearParams = nb.typeof(LinearParams(2, 2))


@nb.njit(FLOAT_2D(FLOAT_2D, _LinearParams))
def linear(x, params):
    out = x @ params.w.T
    np.add(out, params.b.reshape(1, -1), out)
    return out


@nb.experimental.jitclass
class TransformerBlockParams:
    sa: _SelfAttentionParams
    linear1: _LinearParams
    linear2: _LinearParams
    norm1: _LayerNormParams
    norm2: _LayerNormParams

    def __init__(self, emb_dim, n_heads):
        self.sa = SelfAttentionParams(emb_dim, n_heads)
        self.linear1 = LinearParams(emb_dim, emb_dim * 4)
        self.linear2 = LinearParams(emb_dim * 4, emb_dim)
        self.norm1 = LayerNormParams(emb_dim)
        self.norm2 = LayerNormParams(emb_dim)


_TransformerBlockParams = nb.typeof(TransformerBlockParams(4, 2))


@nb.njit(FLOAT_2D(FLOAT_2D, _TransformerBlockParams))
def transformer_block(x, params):
    x = x + self_attention_(layernorm2d_(x.copy(), params.norm1), params.sa)
    x_ = linear(layernorm2d_(x.copy(), params.norm2), params.linear1)
    x = x + linear(gelu(x_, x_), params.linear2)
    return x


@nb.experimental.jitclass
class TransformerParams:
    token_embs: FLOAT_2D
    pos_embs: FLOAT_2D
    norm: _LayerNormParams
    blocks: nb.types.ListType(_TransformerBlockParams)

    def __init__(self, vocab_size, max_seq_len, emb_dim, n_heads, n_layers):
        self.token_embs = np.empty((vocab_size, emb_dim), dtype=FLOAT)
        self.pos_embs = np.empty((max_seq_len, emb_dim), dtype=FLOAT)
        self.norm = LayerNormParams(emb_dim)
        self.blocks = nb.typed.List.empty_list(_TransformerBlockParams)
        for _ in range(n_layers):
            self.blocks.append(TransformerBlockParams(emb_dim, n_heads))


_TransformerParams = nb.typeof(TransformerParams(4, 4, 4, 2, 2))


@nb.njit(FLOAT_2D(LONG_1D, _TransformerParams))
def transformer(x, params):
    tokens = np.empty((x.shape[0], params.token_embs.shape[1]), dtype=FLOAT)
    for i in range(x.shape[0]):
        np.add(params.token_embs[x[i]], params.pos_embs[i], tokens[i])
        layernorm1d_(tokens[i], params.norm)
    for block in params.blocks:
        tokens = transformer_block(tokens, block)
    return tokens


def from_pytorch(nb_module, pt_module):
    if isinstance(nb_module, LayerNormParams):
        nb_module.w[:] = pt_module.weight.detach().numpy()
        nb_module.b[:] = pt_module.bias.detach().numpy()
        nb_module.eps = pt_module.eps

    elif isinstance(nb_module, LinearParams):
        nb_module.w[:] = pt_module.weight.detach().numpy()
        nb_module.b[:] = pt_module.bias.detach().numpy()

    elif isinstance(nb_module, SelfAttentionParams):
        nb_module.w_qkv[:] = pt_module.in_proj_weight.detach().numpy()
        nb_module.w_out[:] = pt_module.out_proj.weight.detach().numpy()
        nb_module.n_heads = pt_module.num_heads

    elif isinstance(nb_module, TransformerBlockParams):
        from_pytorch(nb_module.sa, pt_module.self_attn)
        from_pytorch(nb_module.linear1, pt_module.linear1)
        from_pytorch(nb_module.linear2, pt_module.linear2)
        from_pytorch(nb_module.norm1, pt_module.norm1)
        from_pytorch(nb_module.norm2, pt_module.norm2)

    return nb_module
