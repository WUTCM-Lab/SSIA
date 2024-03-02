import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
import numpy as np
import math


class CrossAttention(nn.Cell):
    def __init__(self):
        super(CrossAttention, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def construct(self, visual, text):
        batch_v = visual.shape[0]
        batch_t = text.shape[0]
        
        visual = ops.Tile()(ops.ExpandDims()(visual, 1), (1, batch_t, 1))
        text = ops.Tile()(ops.ExpandDims()(text, 0), (batch_v, 1, 1))
        
        interaction = ops.Mul()(visual, text)
        text = ops.Mul()(self.sigmoid(interaction), text)
        
        interaction = ops.Mul()(visual, text)
        visual = ops.Mul()(self.sigmoid(interaction), visual)
        
        return text, visual 


if __name__ == '__main__':
    visual = ops.Ones()((60, 256), mindspore.float32)
    text = ops.Ones()((60,256), mindspore.float32)
    print(text.shape)
    print(visual.shape)
    module = CrossAttention()
    text, visual = module(visual, text)
    print(text.shape)
    print(visual.shape)
    print(text)
    print(visual)