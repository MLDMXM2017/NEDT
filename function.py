
import csv
import logging
import os
import sys
import torch

import numpy as np
import torch.nn as nn

from sklearn.metrics import classification_report, roc_auc_score, average_precision_score


def get_device(gpu_id):
    return torch.device("cuda:%d" % gpu_id if torch.cuda.is_available() else "cpu")


def get_dirs(path):
    dir_dict = {
        'res_dir': path,
        'log_dir': "%s/Log" % path,
        'pic_dir': "%s/Picture" % path,
        'mod_dir': '%s/Model' % path,
        'fi_dir': "%s/Feature_Importance" % path,
        'ri_dir': "%s/Rule_Info" % path,
        'pi_dir': "%s/Pred_Info" % path,
        'roc_dir': "%s/ROC" % path,
    }
    for d in dir_dict.values():
        if not os.path.exists(d):
            os.mkdir(d)
    return dir_dict


def get_logger(name, filepath, write=True, output=True):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s -  %(message)s')

    if write:
        file_handler = logging.FileHandler(filepath)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if output:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def get_metric_dict(true, pred, proba):
    report = classification_report(true, pred, output_dict=True)
    performance = {
        'ACC': np.round(report['accuracy'], 4),
        'F1': np.round(report['macro avg']['f1-score'], 4),
        'REC0': np.round(report["0"]['recall'], 4),
        'REC1': np.round(report["1"]['recall'], 4),
        'PRE0': np.round(report["0"]['precision'], 4),
        'PRE1': np.round(report["1"]['precision'], 4),
        'AUC': np.round(roc_auc_score(true, proba[:, 1]), 4),
        'AUPR': np.round(average_precision_score(true, proba[:, 1]), 4),
    }
    return performance


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def output_perf(model, data_set, logger, task_id=0, stage='growth'):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    format_str = "Task {:>2d}, stage {}:  "
    args = [task_id, stage]

    for name, data in data_set.items():
        with torch.no_grad():
            loss, f1 = calc_perf(model, data_set[name].features, data_set[name].targets, criterion)
            format_str += "{}L: {:.4f}  {}F: {:.4f}  ".format(name, loss, name, f1)
            args += [loss, f1]
    output_str = format_str.format(*args)

    logger.info(output_str)


def process_result(logger, res_name, perf_s, met_name, res_path):

    # output result
    logger.info("Final result on dataset {}: ".format(res_name))
    perf_mean = np.round(np.mean(np.array(perf_s), axis=0), 4)
    perf_std = np.round(np.std(np.array(perf_s), axis=0), 4)
    perf_ms = np.array(['{:.4f}\u00B1{:.4f}'.format(perf_mean[i], perf_std[i]) for i in range(len(perf_mean))])
    for i in range(len(met_name)):
        logger.info("{}\tmean:{:.4f}\tstd:{:.4f}".format(met_name[i], perf_mean[i], perf_std[i]))

    # save result
    row_name = np.array([[''] + [str(task_id + 1) for task_id in range(len(perf_s))] +
                         ['mean', 'std', 'mean\u00B1std']]).T
    col_name = np.array([met_name])
    save_data = np.hstack((row_name, np.vstack((col_name, np.vstack((perf_s, perf_mean, perf_std, perf_ms))))))
    with open('{}/{}.csv'.format(res_path, res_name), mode='w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(save_data)


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def calc_impurity(targets, criterion):
    if criterion == 'gini':
        purity = gini_impurity(targets)
    elif criterion == 'entropy':
        purity = entropy(targets)
    else:
        raise TypeError("Unknown criterion!")
    return purity


def gini_impurity(node_y):
    labels, counts = torch.unique(node_y, sorted = True, return_counts = True)
    probs = counts.float() / len(node_y)
    return 1 - torch.sum(probs ** 2)


def entropy(node_y):
    labels, counts = torch.unique(node_y, sorted = True, return_counts = True)
    probs = counts.float() / len(node_y)
    return -torch.sum(probs * torch.log2(probs))


def calc_perf(model, features, targets, criterion):
    outputs = model.forward(features)
    loss = criterion(outputs, targets)
    y_proba = outputs.detach().cpu().numpy()
    y_pred = np.argmax(y_proba, axis=1)
    metric_names, performance = get_metrics(targets.cpu(), y_pred, y_proba)
    return loss, performance[1]


def get_metrics(true, pred, proba):
    metric_names = ['Acc', 'F1', 'Recall_0', 'Recall_1', 'Precision_0', 'Precision_1', 'AUC', 'AUPR']
    report = classification_report(true, pred, output_dict=True)
    acc = np.round(report['accuracy'], 4)
    f1 = np.round(report['macro avg']['f1-score'], 4)
    recall_0 = np.round(report["0"]['recall'], 4)
    recall_1 = np.round(report["1"]['recall'], 4)
    precision_0 = np.round(report["0"]['precision'], 4)
    precision_1 = np.round(report["1"]['precision'], 4)
    auc = np.round(roc_auc_score(true, proba[:, 1]), 4)
    aupr = np.round(average_precision_score(true, proba[:, 1]), 4)
    performance = [acc, f1, recall_0, recall_1, precision_0, precision_1, auc, aupr]
    return metric_names, performance


def output_node_info(logger, node):
    if node.is_leaf:
        logger.info("leaf id: {}".format(node.node_id))
        format_str = "depth: {:<3d}    sample_num: {:<4d}    class_dist: {:>4d}:{:<4d} "
        format_str += "    prob_dist: "
        format_str += ":".join(["{:.4f}"] * node.n_class)
        format_str += "    path_prob: {:.4f}    impurity: {:.4f}    reason: {}"
        args = [node.depth, node.sample_num]
        args += node.class_dist.tolist()
        args += node.prob_dist.tolist()
        args += [node.path_prob, node.impurity, node.reason]
        output_str = format_str.format(*args)
        logger.info(output_str)
    else:
        logger.info("node id: {}".format(node.node_id))
        format_str = "depth: {:<3d}    sample_num: {:<4d}    class_dist: {:>4d}:{:<4d} "
        format_str += "    prob_dist: "
        format_str += ":".join(["{:.4f}"] * node.n_class)
        format_str += "    path_prob: {:.4f}    impurity: {:.4f}    reason: {}"
        args = [node.depth, node.sample_num]
        args += node.class_dist.tolist()
        args += node.prob_dist.tolist()
        args += [node.path_prob, node.impurity, node.reason]
        output_str = format_str.format(*args)
        logger.info(output_str)
















