from collections import OrderedDict
import shutil
import numpy as np
import sys
import math

import mindspore
from mindspore import ops
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import Parameter
from layers.loss import cosine_similarity

def log_to_txt( contexts=None,filename="save.txt", mark=False,encoding='UTF-8',mode='a'):
    f = open(filename, mode,encoding=encoding)
    if mark:
        sig = "------------------------------------------------\n"
        f.write(sig)
    elif isinstance(contexts, dict):
        tmp = ""
        for c in contexts.keys():
            tmp += str(c)+" | "+ str(contexts[c]) +"\n"
        contexts = tmp
        f.write(contexts)
    else:
        if isinstance(contexts,list):
            tmp = ""
            for c in contexts:
                tmp += str(c)
            contexts = tmp
        else:
            contexts = contexts + "\n"
        f.write(contexts)

    f.close()

def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]
    return dict_to

def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count

def ct(score, margin):
    a = -ops.Exp()(5 * score) + ops.Exp()(Tensor([5], mindspore.float32))
    b = ops.Exp()(Tensor([5], mindspore.float32)) - 1
    c = (a / b) * margin
    return c

def load_from_txt(filename, encoding="utf-8"):
    f = open(filename,'r' ,encoding=encoding)
    contexts = f.readlines()
    return contexts

def acc_i2t2(input_array):
    # 假设每个图像对应5个文本
    image_size = input_array.shape[0] // 5
    ranks = np.zeros(image_size)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        # 对于每个图像，找出与之匹配的文本的排名
        inds = np.argsort(input_array[5 * index:5 * index + 5].flatten())[::-1]
        rank = 1e20
        # 在这里，我们只需要考虑当前图像对应的5个文本
        for i in range(5):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)

def acc_t2i2(input_array):
    image_size = input_array.shape[0] // 5  # 假设每个图像对应5个文本
    ranks = np.zeros(image_size)
    top1 = np.zeros(image_size)

    for index in range(image_size):
        if 5 * index + 4 < input_array.shape[0]:
            inds = np.argsort(input_array[5 * index:5 * index + 5].flatten())[::-1]
            rank = 1e20
            for i in range(5):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            top1[index] = inds[0]
        else:
            remaining = input_array.shape[0] - 5 * index
            inds = np.argsort(input_array[5 * index:5 * index + remaining].flatten())[::-1]
            rank = 1e20
            for i in range(remaining):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            top1[index] = inds[0]

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1, r5, r10, medr, meanr), (ranks, top1)


def shard_dis(images, captions, model, shard_size=50):

    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1

    d = np.zeros((len(images), len(captions)))

    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            
            im = images[im_start:im_end]
            s = captions[cap_start:cap_end]

            Ft, mvsa_feature, mvsa_feature_nt, text_feature = model(im, s)
            sim = cosine_similarity(mvsa_feature, Ft)
            
            d[im_start:im_end, cap_start:cap_end] = sim.asnumpy()
        # print(f"total:{n_im_shard}, now: {i}")
    return d


