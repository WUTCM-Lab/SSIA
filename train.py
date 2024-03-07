import utils
from data import get_loaders
import engine
import os,random,copy
import argparse
import yaml
import logging
import mindspore
from layers.AMFMN import factory
import mindspore.nn as nn
from mindspore import context
from engine import NetwithLoss, AverageMeter

def generate_random_samples(options):
    # load all anns
    auds = utils.load_from_txt(options['dataset']['data_path']+'train_auds.txt')
    fnames = utils.load_from_txt(options['dataset']['data_path']+'train_filename.txt')

    # merge
    assert len(auds) // 5 == len(fnames)
    all_infos = []
    
    for img_id in range(len(fnames)):
        aud_id = [img_id * 5 ,(img_id+1) * 5]
        all_infos.append([auds[aud_id[0]:aud_id[1]], fnames[img_id]])

    # shuffle
    random.shuffle(all_infos)

    # split_trainval
    percent = 0.8
    train_infos = all_infos[:int(len(all_infos)*percent)]
    val_infos = all_infos[int(len(all_infos)*percent):]

    # save to txt
    train_auds = []
    train_fnames = []
    for item in train_infos:
        for aud in item[0]:
            train_auds.append(aud)
        train_fnames.append(item[1])
    utils.log_to_txt(train_auds, options['dataset']['data_path']+'train_auds_verify.txt',mode='w')
    utils.log_to_txt(train_fnames, options['dataset']['data_path']+'train_filename_verify.txt',mode='w')

    val_auds = []
    val_fnames = []
    for item in val_infos:
        for aud in item[0]:
            val_auds.append(aud)
        val_fnames.append(item[1])
    utils.log_to_txt(val_auds, options['dataset']['data_path']+'val_auds_verify.txt',mode='w')
    utils.log_to_txt(val_fnames, options['dataset']['data_path']+'val_filename_verify.txt',mode='w')

    print("Generate random samples to {} complete.".format(options['dataset']['data_path']))

def update_options_savepath(options, k):
    updated_options = copy.deepcopy(options)

    updated_options['k_fold']['current_num'] = k
    updated_options['logs']['ckpt_save_path'] = options['logs']['ckpt_save_path'] + \
                                                options['k_fold']['experiment_name'] + "/" + str(k) + "/"
    return updated_options

def parser_options():
    # Hyper Parameters setting
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default='option/RSICD_AMFMN.yaml', type=str,
                         help='path to a yaml options file')
    opt = parser.parse_args()

    # load model options
    with open(opt.path_opt, 'r') as handle:
        options = yaml.load(handle,Loader=yaml.FullLoader)

    return options

def train(options):
    
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

    # make ckpt save dir
    if not os.path.exists(options['logs']['ckpt_save_path']):
        os.makedirs(options['logs']['ckpt_save_path'])
    
    # Create dataset, model, criterion and optimizer
    train_loader, val_loader = get_loaders(options)
    
    f_model = factory(options)
    
    optimizer = nn.Adam(params=f_model.trainable_params(), learning_rate=options['optim']['lr'])
    
    if options['optim']['resume']:
        pass
    else:
        start_epoch = 0
    # Train the Model
    best_rsum = 0
    best_score = ""
    
    model = NetwithLoss(f_model, options['optim']['margin'], options['dataset']['batch_size'])

    model = nn.TrainOneStepCell(model, optimizer)
    
    loss_meter = AverageMeter('loss')
    
    for epoch in range(start_epoch, options['optim']['epochs']):
        engine.train(train_loader, model, epoch, loss_meter, options)
        if not epoch % options['logs']['eval_step']:
            rsum, all_scores = engine.validate(val_loader, f_model)
            
            is_best = rsum >= best_rsum
            best_rsum = rsum if is_best else best_rsum
            best_score = all_scores if is_best else best_score
            
            print(f"Current {options['k_fold']['current_num']}th fold.")
            print(f"Best score: {best_score}")
            print(f"Now  score: {all_scores}")
            mindspore.save_checkpoint(model, options['logs']['ckpt_save_path'] + options['model']['name'] + "_best.ckpt")
        loss_meter.reset()
 
def main():
    options = parser_options()

    # k_fold verify
    for k in range(options['k_fold']['nums']):
        print("=========================================")
        print("Start {}th fold".format(k))

        # generate random train and val samples
        generate_random_samples(options)

        # update save path
        update_options = update_options_savepath(options, k)

        # run experiment
        train(update_options)

if __name__ == '__main__':
    main()
