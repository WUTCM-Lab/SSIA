import numpy as np
from math import sqrt
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import initializer, Uniform

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = ops.Pow()(X, 2).sum(axis=dim, keep_dims=True).sqrt() + eps
    X = ops.Div()(X, norm)
    return X

def qkv_attention(query, key, value, mask=None, dropout=None):
    d_k = query.shape[-1]
    scores = ops.matmul(query, ops.swapaxes(key, -1, -2)) / sqrt(d_k)
    p_attn = ops.softmax(scores, axis=-1)
    return ops.matmul(p_attn, value)

class GatedFusion(nn.Cell):
    def __init__(self, dim, num_attn, dropout=0.01, fusion_func="sum"):
        super(GatedFusion, self).__init__()
        self.dim = dim

        self.fusion_func = fusion_func

        in_dim = dim * 2 if fusion_func == "concat" else dim
        self.fc_1 = nn.SequentialCell([
            nn.Dense(in_dim, dim, weight_init=Uniform(scale=sqrt(6. / (dim + dim)))),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])

    def construct(self, v1, v2, mask=None):
        k1 = v1
        k2 = v2
        batch_size_v1 = v1.shape[0]
        batch_size_v2 = v2.shape[0]

        v1 = ops.Tile()(ops.ExpandDims()(v1, 1), (1, batch_size_v2, 1, 1))
        k1 = ops.Tile()(ops.ExpandDims()(k1, 1), (1, batch_size_v2, 1, 1))
        v2 = ops.Tile()(ops.ExpandDims()(v2, 0), (batch_size_v1, 1, 1, 1))
        k2 = ops.Tile()(ops.ExpandDims()(k2, 0), (batch_size_v1, 1, 1, 1))
        
        weighted_v2 = qkv_attention(k1, k2, v2)
        gate_v1 = ops.Sigmoid()((v1 * weighted_v2).sum(axis=-1)).unsqueeze(-1)

        if self.fusion_func == "sum":
            fused_v1 = (k1 + weighted_v2) * gate_v1
        elif self.fusion_func == "concat":
            fused_v1 = ops.Concat(-1)((k1, weighted_v2)) * gate_v1

        return self.fc_1(fused_v1) + v1


if __name__ == '__main__':
    a=ops.ones((4,5,3))
    b=ops.zeros((4,5,3))

    c=GatedFusion(3,1)(a,b)
    print(c.shape)
    print(c)