import engine
import argparse
import yaml
import utils
from data import get_test_loader
import engine
import os,random,copy
import argparse
import yaml
import logging
import mindspore
from layers.AMFMN import factory
from mindspore import context
import numpy as np

def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/RSITMD_AMFMN.yaml', type=str,
                        help='path to a yaml options file')
    opt = parser.parse_args()

    # load model options
    with open(opt.path_opt, 'r') as handle:
        options = yaml.load(handle,Loader=yaml.FullLoader)

    return options

def update_options_savepath(options, k):
    updated_options = copy.deepcopy(options)

    updated_options['optim']['resume'] = options['logs']['ckpt_save_path'] + options['k_fold']['experiment_name'] + "/" \
                                         + str(k) + "/" + options['model']['name'] + '_best.ckpt'

    return updated_options


def test(options):

    
    test_loader = get_test_loader(options)
    
    model = factory(options)
    
    if os.path.isfile(options['optim']['resume']):
        param_dict = mindspore.load_checkpoint(options['optim']['resume'])
        mindspore.load_param_into_net(model, param_dict)
    else:
        print("=> no checkpoint found at '{}'".format(options['optim']['resume']))
    
    rsum, all_scores = engine.validate(test_loader, model)
    print(all_scores)
    
    
if __name__ == '__main__':
    options = parser_options()
    
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    
    # calc ave k results
    last_score = []
    
    for k in range(options['k_fold']['nums']):
        print("=========================================")
        print("Start evaluate {}th fold".format(k))

        # update save path
        update_options = update_options_savepath(options, k)

        # run experiment
        one_score = test(update_options)
        last_score.append(one_score)
        
        print("Complete evaluate {}th fold".format(k))

    print("\n===================== Ave Score ({}-fold verify) =================".format(options['k_fold']['nums']))
    last_score = np.average(last_score, axis=0)
    names = ['r1i', 'r5i', 'r10i', 'r1t', 'r5t', 'r10t', 'mr']
    for name,score in zip(names,last_score):
        print("{}:{}".format(name, score))
    print("\n==================================================================")