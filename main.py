# -*- coding: gbk -*-

import datetime
import os
import pickle
import sys

import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold

from data_tool import load_data, process_data
from function import get_device, get_dirs, get_logger, get_metric_dict, make_dir, output_perf, process_result, set_seed
from NeuralDecisionTree import NeuralDecisionTree

sys.setrecursionlimit(10000)

# Experimental parameters
n_repeat = 3        # Repeat Experiment
n_fold = 5          # k-fold cross valiadation
random_state = 42   # Random seed
n_process = n_repeat * n_fold

# Cuda parameters
use_cuda = True
gpu_names = [0]
n_gpu = len(gpu_names)

# Datasets
file_names = ['DC_exp', 'SL_exp']

# Data directory
ori_dir = 'Datasets'
dat_dir = make_dir("data")
src_dir = make_dir("%s/source_data" % dat_dir)

# Result directory
res_path = make_dir("Results")
log_name = '%s/main.txt' % res_path
LOGGER = get_logger('main', log_name)


def do_exp(task_id, data, indice, file_name, dir_dict):

    LOGGER.info("Run task {:>2d}, pid is {}...".format(task_id, os.getpid()))
    start = datetime.datetime.now()

    set_seed(random_state)
    device = get_device(gpu_names[task_id % n_gpu])

    # Prepare data
    data_set = process_data(data, indice, random_state, device, spl_num=3)

    # Train model
    model = NeuralDecisionTree(file_name=file_name, exp_id=task_id, device=device, dir_dict=dir_dict,
                               random_state=random_state)
    model.cuda()

    # Greedy growth stage
    stage = 'growth'
    LOGGER.info("Task {:>2d}, stage {} start...".format(task_id, stage))
    model_name = f"{dir_dict['mod_dir']}/model_{task_id}_{stage}.dat"
    model.fit(data_set['Trn'])
    pickle.dump(model, open(model_name, "wb"))
    output_perf(model, data_set, LOGGER, task_id, stage)
    # feature_importance = model.get_feature_importance()
    # np.savetxt(f"{dir_dict['fi_dir']}/fi_{task_id}_{stage}.csv", feature_importance, fmt='%s', delimiter=',')

    # Global fine-tuning stage
    stage = 'refine'
    LOGGER.info("Task {:>2d}, stage {} start...".format(task_id, stage))
    model_name = f"{dir_dict['mod_dir']}/model_{task_id}_{stage}.dat"
    model.refine(data_set)
    pickle.dump(model, open(model_name, "wb"))
    output_perf(model, data_set, LOGGER, task_id, stage)
    feature_importance = model.get_feature_importance()
    np.savetxt(f"{dir_dict['fi_dir']}/fi_{task_id}_{stage}.csv", feature_importance, fmt='%s', delimiter=',')

    pickle.dump(model, open(f"{dir_dict['mod_dir']}/model_{task_id}.dat", "wb"))

    end = datetime.datetime.now()
    time_cost = end - start
    LOGGER.info("Task {:>2d}, finished! Cost time: {}".format(task_id, time_cost))

    # Performance on test data
    outputs = model.forward(data_set['Tst'].features, data_set['Trn'])
    y_proba = outputs.detach().cpu().numpy()
    y_pred = np.argmax(y_proba, axis=1)
    perf_dict = get_metric_dict(data_set['Tst'].targets.cpu().numpy(), y_pred, y_proba)
    output_str = "Task {:>2d} on TestSet:  ".format(task_id)
    for met, perf in perf_dict.items():
        if met not in ['REC0', 'PRE0']:
            output_str += "{}={:.4f}  ".format(met, perf)
    LOGGER.info(output_str)

    del model, data_set

    return list(perf_dict.keys()), list(perf_dict.values())


if __name__ == '__main__':

    for file_name in file_names:
        LOGGER.info("########################################################################################")
        LOGGER.info("DataSet: {}".format(file_name))

        dir_dict = get_dirs(f"{res_path}/{file_name}")

        # Load data
        data_path = '%s/%s.csv' % (src_dir, file_name)
        if not os.path.exists(data_path):
            data = load_data(ori_dir, file_name)
            np.savetxt(data_path, data, fmt='%s', delimiter=',')
        else:
            data = np.loadtxt(data_path, float, delimiter=',')
        x, y = data[:, :-1], data[:, -1].astype(int)

        # Perform three 5-fold cross-validation
        skf = RepeatedStratifiedKFold(n_splits=n_fold, n_repeats=n_repeat, random_state=random_state)
        indices = [(train_id, test_id) for train_id, test_id in skf.split(x, y)]

        # # Multi-process
        # LOGGER.info("Assign tasks for process pool...")
        # pool = mp.Pool()
        # tasks = [pool.apply_async(do_exp, args=(task_id, data, indices[task_id], file_name, dir_dict))
        #          for task_id in range(n_process)]
        # pool.close()
        # pool.join()
        # results = [task.get() for task in tasks]

        # Single-process
        LOGGER.info("Perform experiments for each task...")
        results = [do_exp(task_id, data, indices[task_id], file_name, dir_dict) for task_id in range(n_process)]

        # Collate experimental results
        perf_s = []
        for task_id in range(n_process):
            met_name, perf = results[task_id]
            perf_s.append(perf)

        process_result(LOGGER, f"{file_name}", perf_s, met_name, res_path)





