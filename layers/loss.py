import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops import functional as F
from mindspore import dtype as mstype
import mindspore.context as context
from mindspore import Tensor
import numpy as np
import mindspore.numpy as mnp

def cosine_similarity(x1, x2, dim=-1, eps=1e-8):
    w12 = ops.ReduceSum()(ops.Mul()(x1, x2), dim)
    w1 = ops.norm(x1, 2, dim)
    w2 = ops.norm(x2, 2, dim)
    return (w12 / ops.clip_by_value(w1 * w2, clip_value_min=eps, clip_value_max=None)).squeeze()

class NTXentLoss(nn.Cell):

    def __init__(self, batch_size, temperature=0.1, use_cosine_similarity=True, alpha_weight=0.75):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.alpha_weight = alpha_weight
        self.softmax = nn.Softmax(axis=-1)
        self.criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='sum')

    def softXEnt(self, target, logits):
        logprobs = ops.LogSoftmax(axis=-1)(logits)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def construct(self, zis, zjs, norm=True, weights=1.0):
        temperature = self.temperature
        alpha = self.alpha_weight

        if norm:
            zis = ops.L2Normalize(axis=-1, epsilon=1e-12)(zis)
            zjs = ops.L2Normalize(axis=-1, epsilon=1e-12)(zjs)

        hidden1, hidden2 = zis, zjs # torch.Size([60, 256])
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2
        labels = ops.one_hot(mnp.arange(0, batch_size, dtype=mnp.int64), batch_size, Tensor(1.0, ms.float32), Tensor(0.0, ms.float32))
        masks = ops.one_hot(mnp.arange(0, batch_size, dtype=mnp.int64), batch_size)

        logits_ab = ops.MatMul()(hidden1, ops.Transpose()(hidden2_large, (1, 0))) / temperature
        logits_ba = ops.MatMul()(hidden2, ops.Transpose()(hidden1_large, (1, 0))) / temperature

        loss_a = self.softXEnt(labels, logits_ab)
        loss_b = self.softXEnt(labels, logits_ba)

        return alpha * loss_a + (1 - alpha) * loss_b

class CalculateLoss(nn.Cell):
    def __init__(self, size, margin):
        super(CalculateLoss, self).__init__()
        self.size = size
        self.margin = margin
        
    def construct(self, scores):
        diagonal = ops.ExpandDims()(ops.DiagPart()(scores), 1)

        d1 = ops.BroadcastTo(scores.shape)(diagonal) # 60, 60
        d2 = ops.Transpose()(d1, (1, 0)) # 60 60
        
        # cost_s = ops.clip_by_value(self.margin + scores - d1, 0, None)
        # cost_im = ops.clip_by_value(self.margin + scores - d2, 0, None)
        cost_s = ops.maximum(self.margin + scores - d1, ops.zeros_like(scores))
        cost_im = ops.maximum(self.margin + scores - d2, ops.zeros_like(scores))

        mask = mnp.eye(scores.shape[0]) > 0.5
        cost_s = ops.MaskedFill()(cost_s, mask, 0.0)
        cost_im = ops.MaskedFill()(cost_im, mask, 0.0)
        
        loss = cost_s.sum() + cost_im.sum()
        
        return loss

class TotalLoss(nn.LossBase):
    def __init__(self, size, margin, reduction='none'):
        super(TotalLoss, self).__init__(reduction)
        self.nt = NTXentLoss(128)
        self.loss = CalculateLoss(size, margin)
        
    def construct(self, Ft, mvsa_feature, mvsa_feature_nt, text_feature):
        score = cosine_similarity(mvsa_feature, Ft)
        loss_nt = self.nt(mvsa_feature_nt, text_feature)
        loss = self.loss(score)
        return loss + 0.1 * loss_nt
        
if __name__ == '__main__':
    # mvsa :torch.Size([60, 256])
    # text_feature : torch.Size([60, 256])
    zis = Tensor(np.random.rand(60, 256), ms.float32)
    zjs = Tensor(np.random.rand(60, 256), ms.float32)
    loss_fn = NTXentLoss(128)
    loss = loss_fn(zis, zjs)
    print(loss)
