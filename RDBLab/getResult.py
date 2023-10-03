import numpy as np
import torch

from RDBLab.build_dataset import build_ds
from RDBLab.build_model import build_md
from RDBLab.utils.opt_parser import getParser

version = 'RDBLab/2200-2000-2000-2000/tpch'
dataset_ = 'PSQLTPCH'
new_ds = True
new_md = True
data_dir = 'tpch'


def getConfigResult(config):
    config = str(config)
    opt = getParser(version=version,
                    dataset=dataset_,
                    new_ds=new_ds,
                    new_md=new_md,
                    mid_data_dir=version + '/' + config,
                    data_structure=version + '/data_structure' + config,
                    data_dir=data_dir + "/" + config,
                    saved_model='/save_model_QPPNet',
                    mode='config_eval').parse_args(args=[])

    dataset, dim_dict = build_ds(opt, 'config_model')
    pred_times = build_md(dataset, 'QPPNet', opt, dim_dict)

    return pred_times


def compare(config1, config2):
    pred_times1 = getConfigResult(config1)
    pred_times2 = getConfigResult(config2)

    pred_timess1 = [time.cpu().detach().numpy() for time in pred_times1]
    pred_timess2 = [time.cpu().detach().numpy() for time in pred_times2]

    total_times1 = []
    total_times2 = []
    for idx, pt in enumerate(pred_timess2):
        tempTT = pred_timess1[idx]
        tempPT = pt
        total_times1.extend(list(tempTT))
        total_times2.extend(list(tempPT))

    sum1 = sum(total_times1) / 1000
    sum2 = sum(total_times2) / 1000

    if sum1 > sum2:
        prio = config2
        diff = sum1 - sum2
        inference = diff / sum1
    else:
        prio = config1
        diff = sum2 - sum1
        inference = diff / sum2

    return sum1, sum2, prio, round(diff,2), round(inference,4)*100


def get_result(a,b):
    sum1, sum2, prio, diff, inference = compare(a,b)
    lines = "Config{} approximate time cost {}s.\n\n" \
            "Config{} approximate time cost {}s.\n\n" \
            "So Config{} runs faster.\n\n" \
            "It is {} seconds faster than the other config.\n\n" \
            "With throughput {}% higher".format(a,sum1,b, sum2, prio, diff, inference)

    return lines
