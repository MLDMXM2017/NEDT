
import torch
import warnings

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_tool import calc_entropy
from Gate import Gate

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)


class NeuralDecisionNode(nn.Module):

    def __init__(self,
                 node_id='Unknown', device=torch.device("cpu"), criterion='entropy',
                 n_class=2, use_cuda=False):

        self.node_id = node_id
        self.device = device
        self.criterion = criterion

        self.n_class = n_class
        self.use_cuda = use_cuda

        super(NeuralDecisionNode, self).__init__()
        self.dtype = torch.FloatTensor

        self.depth = None
        self.sample_num = None
        self.class_dist = None
        self.path_prob = 1
        self.prob_dist = None
        self.impurity = None
        self.class_prob = None

        self.is_leaf = True

        self.reason = None

        self.gate = None
        self.left_child = None
        self.right_child = None

        self.best_epoch = None

        # torch.autograd.set_detect_anomaly(True)

    def cuda(self):
        self.use_cuda = True
        self.dtype = torch.cuda.FloatTensor
        super(NeuralDecisionNode, self).to(self.device)

    def foreach(self, func):
        func(self)
        if not self.is_leaf:
            self.left_child.foreach(func)
            self.right_child.foreach(func)

    def gating(self, feature):
        return self.gate(feature).clamp(min=0.00001, max=0.99999)

    def init_split(self, data_info):

        self.gate = Gate(self.device, data_info)
        if self.use_cuda:
            self.gate.cuda()

        self.left_child = NeuralDecisionNode(node_id=self.node_id + 'l', device=self.device, criterion=self.criterion,
                                             n_class=self.n_class, use_cuda=self.use_cuda)
        self.right_child = NeuralDecisionNode(node_id=self.node_id + 'r', device=self.device, criterion=self.criterion,
                                              n_class=self.n_class, use_cuda=self.use_cuda)

    def refine(self, train_set):

        self.train()
        self.change_param_state('layer1', True)
        self.change_param_state('layer2', True)

        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=0.0)

        max_epoch = 500
        patience = 50
        best_loss = float('inf')
        best_params = None
        best_epoch = max_epoch
        counter = 0

        for epoch in range(1, max_epoch + 1):

            trn_loss = self.calc_loss_entropy(train_set)

            l2_reg = torch.tensor(0., dtype=torch.float32).to(self.device)
            for name, param in self.named_parameters():
                if "layer2.weight" in name:
                    l2_reg += torch.sum(torch.square(param))
            loss = trn_loss + 0.01 * l2_reg

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_params = self.state_dict()
                best_epoch = epoch
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        self.load_state_dict(best_params)
        self.best_epoch = best_epoch

    def change_param_state(self, key, update):
        for name, param in self.named_parameters():
            if key in name:
                if update:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def calc_loss_entropy(self, train_set):

        prob = self.gating(train_set.features)

        left_sample_prob = 1.0 - prob
        right_sample_prob = prob

        left_weight = left_sample_prob.mean()
        right_weight = right_sample_prob.mean()

        one_hot_target = torch.nn.functional.one_hot(train_set.targets, num_classes=self.n_class).float()

        left_sample_weight = left_sample_prob / left_sample_prob.sum()
        right_sample_weight = right_sample_prob / right_sample_prob.sum()

        left_prob_dist = (left_sample_weight.view(-1, 1) * one_hot_target).sum(dim=0)
        right_prob_dist = (right_sample_weight.view(-1, 1) * one_hot_target).sum(dim=0)

        left_entropy = calc_entropy(left_prob_dist, self.device)
        right_entropy = calc_entropy(right_prob_dist, self.device)

        return left_weight * left_entropy + right_weight * right_entropy

    def split(self, data_set):

        self.eval()

        prob = self.gating(data_set.features)

        left_idx = (1.0 - prob).round().nonzero().flatten()
        right_idx = prob.round().nonzero().flatten()

        return left_idx, right_idx

    def forward(self, features, sup_set=None, sup_prob=None):

        if self.is_leaf:

            if sup_set is not None:
                sup_weight = sup_prob / sup_prob.sum()
                one_hot_target = torch.nn.functional.one_hot(sup_set.targets, num_classes=self.n_class).float()
                self.prob_dist = (sup_weight.view(-1, 1) * one_hot_target).sum(dim=0)

            return self.prob_dist

        else:
            prob = self.gating(features)

            left_router_prob = 1.0 - prob
            right_router_prob = prob

            left_sup_prob, right_sup_prob = None, None
            if sup_set is not None:
                tmp_prob = self.gating(sup_set.features)
                left_sup_prob = (1.0 - tmp_prob) * sup_prob
                right_sup_prob = tmp_prob * sup_prob

            return left_router_prob.unsqueeze(1) * self.left_child(features, sup_set, left_sup_prob) + \
                   right_router_prob.unsqueeze(1) * self.right_child(features, sup_set, right_sup_prob)


