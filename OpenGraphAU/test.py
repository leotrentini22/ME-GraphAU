import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader #this changes
from tqdm import tqdm #this changes
import logging #this changes

from model.ANFL import MEFARG
from dataset import * #this changes
from utils import *
from conf import get_config,set_logger,set_outdir,set_env


def get_dataloader(conf):  #this changes -> now we need to take data from dataset
    print('==> Preparing data...')
    testset = AffWild2(conf.dataset_path, phase='test', transform=image_eval(crop_size=conf.crop_size),
                             stage=1)
    test_loader = DataLoader(testset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers)
    return test_loader, len(testset)



#Val
def test(net, test_loader):
    net.eval()
    statistics_list = None
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        targets = targets.float()
        with torch.no_grad():
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            # outputs, _ = net(inputs)
            outputs = net(inputs)
            outputs_filtered = outputs[:, [0,1,2,4,5,7,9,12,19,20,21,22]]  #I have only particular AUs
            update_list = statistics(outputs_filtered, targets.detach(), 0.5)
            statistics_list = update_statistics_list(statistics_list, update_list)
    mean_f1_score, f1_score_list = calc_f1_score(statistics_list)
    mean_acc, acc_list = calc_acc(statistics_list)
    return mean_f1_score, f1_score_list, mean_acc, acc_list


# COSE DE FARE
# NN rida 41 categorie -> selezionare solo quelle che ci interessano

def main(conf):
    dataset_info = AffWild2_infolist  # function in 'utils', different from 'demo' because we don't need to output AUs, we need to eval accuracy

    # data     
    test_loader, test_data_num = get_dataloader(conf)
    logging.info("[ test_data_num: {} ]".format( test_data_num))
    net = MEFARG() #num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc, neighbor_num=conf.neighbor_num, metric=conf.metric)

    # resume
    if conf.resume != '':
        logging.info("Resume form | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)

    if torch.cuda.is_available():
        net = nn.DataParallel(net).cuda()

    #test
    test_mean_f1_score, test_f1_score, test_mean_acc, test_acc = test(net, test_loader)

    # log
    infostr = {'test_mean_f1_score {:.2f} test_mean_acc {:.2f}' .format(100.* test_mean_f1_score, 100.* test_mean_acc)}
    logging.info(infostr)
    infostr = {'F1-score-list:'}
    logging.info(infostr)
    infostr = dataset_info(test_f1_score)
    logging.info(infostr)
    infostr = {'Acc-list:'}
    logging.info(infostr)
    infostr = dataset_info(test_acc)
    logging.info(infostr)



# ---------------------------------------------------------------------------------


if __name__=="__main__":
    conf = get_config()
    conf.evaluate = True
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)

