import time
import logging
import utils
import mindspore.nn as nn
import mindspore
from mindspore import context
from mindspore import Tensor
from mindspore.train.callback import Callback
import mindspore.ops as ops
from mindspore import Model, ParameterTuple
from mindspore.train.callback import TimeMonitor, LossMonitor
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
import numpy as np
from layers.loss import TotalLoss
from utils import shard_dis
import mindspore.numpy as mnp


class TrainOneEpoch(nn.Cell):
    def __init__(self, network, optimizer, grad_clip, margin, size):
        super(TrainOneEpoch, self).__init__(auto_prefix=False)
        self.network = network
        self.loss_fn = TotalLoss(size=size, margin=margin)
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.grad_operation = ops.GradOperation(get_by_list=True)
        self.weights = ParameterTuple(network.trainable_params())

    def construct(self, images, captions):

        predictions = self.network(images, captions)
        loss = self.loss_fn(*predictions)  
        
        grads = self.grad_operation(self.loss_fn, self.weights)(*predictions) 

        if self.grad_clip is not None:
            grads = ops.clip_by_global_norm(grads, self.grad_clip)

        self.optimizer(grads)
        return loss
    
def train(train_loader, model, optimizer, epoch, opt={}):
    train_network = TrainOneEpoch(model, optimizer, opt['optim']['grad_clip'], opt['optim']['margin'], opt['dataset']['batch_size'])
    model.set_train(True)
    for i, data in enumerate(train_loader.create_dict_iterator()):
        start = time.time()
        loss = train_network(data["image"], data["audio"])
        end = time.time()
        if not i % opt['logs']['print_freq']:
            print(f"Epoch: {epoch}, Step: {i}, Loss: {loss}")
            print(f"Time per step: {end - start} seconds")


def validate(val_loader, model):
    model.set_train(False)
    start = time.time()
    dataset_len = val_loader.children[0].source_len
    input_visual = mnp.zeros((dataset_len, 3, 224, 224), dtype=mindspore.float32)
    input_text = mnp.zeros((dataset_len, 1, 300, 64), dtype=mindspore.float32)
    ids = 0

    for val_data in val_loader:
        imgs, caps = val_data  
        batch_size = imgs.shape[0]  

        input_visual[ids:ids+batch_size] = imgs
        input_text[ids:ids+batch_size] = caps

        ids += batch_size

    input_visual = input_visual[::5]
    
    d = shard_dis(input_visual, input_text, model)
    end = time.time()
    print("calculate similarity time:", end - start)
    (r1i, r5i, r10i, medri, meanri), _ = utils.acc_i2t2(d)
    print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanri))
    (r1t, r5t, r10t, medrt, meanrt), _ = utils.acc_t2i2(d)
    print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1t, r5t, r10t, medrt, meanrt))
    currscore = (r1t + r5t + r10t + r1i + r5i + r10i)/6.0

    all_score = "r1i:{} r5i:{} r10i:{} medri:{} meanri:{}\n r1t:{} r5t:{} r10t:{} medrt:{} meanrt:{} sum:{}\n".format(
        r1i, r5i, r10i, medri, meanri, r1t, r5t, r10t, medrt, meanrt, currscore
    )
    return currscore, all_score