
import math
import torch

import numpy as np

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from Datasets.DC.data_loader import load_data as load_data_DC
from Datasets.SL.data_loader_sl import read_data as load_data_SL


class MyDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        self.n_sample, self.n_feature, self.n_pair = self.features.shape

        labels, counts = torch.unique(targets, return_counts=True)
        self.n_class = len(labels)
        self.class_label = labels
        self.class_weight = {label: self.n_sample / count for label, count in zip(labels, counts)}

    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.targets[idx]
        return idx, x, y

    def __len__(self):
        return len(self.targets)

    @property
    def shape(self):
        data_shape = {
            'feature': self.features.shape,
            'target': self.targets.shape
        }
        return data_shape

    @property
    def class_dist(self):
        counts = torch.bincount(self.targets)
        class_dist = {i.item(): counts[i].item() for i in torch.nonzero(counts)}
        return class_dist

    def cuda(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)


def load_data(in_dir, file_name):
    data_name, feature_comb = file_name.split('_')
    feature_types = feature_comb.split('+')
    if data_name == 'DC':
        return load_data_DC(in_dir, feature_types=feature_types)
    elif data_name == 'SL':
        return load_data_SL(in_dir, feature_types=feature_types)


def process_data(data, indice, random_state=42, device=None, task_type='clf', spl_num=2):

    x, y = to_pairing_vec(data[:, :-1]), data[:, -1].astype(int)

    x_train, y_train = x[indice[0]], y[indice[0]]
    x_test, y_test = x[indice[1]], y[indice[1]]

    label_type = torch.int64 if task_type == 'clf' else torch.float32

    if spl_num == 2:
        data_set = {
            'Trn': package_data(x_train, y_train, label_type, device),
            'Tst': package_data(x_test, y_test, label_type, device)
        }
    elif spl_num == 3:
        x_trn, x_vld, y_trn, y_vld = train_test_split(x_train, y_train, test_size=0.25, random_state=random_state,
                                                      shuffle=True, stratify=y_train)
        data_set = {
            'Trn': package_data(x_trn, y_trn, label_type, device),
            'Vld': package_data(x_vld, y_vld, label_type, device),
            'Tst': package_data(x_test, y_test, label_type, device)
        }
    else:
        raise TypeError("Error spl_num!")

    return data_set


def to_pairing_vec(x, pair_num=2):
    n_pair = x.shape[1] // pair_num
    paired_x = np.zeros((x.shape[0], n_pair, pair_num))
    for i in range(pair_num):
        paired_x[:, :, i] = x[:, i*n_pair:(i+1)*n_pair]
    return paired_x


def package_data(x, y, label_type, device=None):
    tensor_x = torch.tensor(x, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=label_type)
    dataset = MyDataset(tensor_x, tensor_y)
    if device is not None:
        dataset.cuda(device)
    return dataset


def calc_entropy(probs, device = None):
    entropy = torch.tensor(0.0)
    if device is not None:
        entropy = entropy.to(device)

    for p in probs:
        if p > 0:
            entropy -= p * math.log(p, 2)

    return entropy




