
import torch
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

from data_tool import package_data
from function import calc_impurity, calc_perf, get_logger, output_node_info
from NeuralDecisionNode import NeuralDecisionNode
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)


class NeuralDecisionTree(nn.Module):
    def __init__(self,
                 file_name='Unknown', exp_id=-1, device=torch.device("cpu"), dir_dict=None,
                 criterion='entropy', max_depth=None,
                 need_visual=True, random_state=42):

        self.file_name = file_name
        self.exp_id = exp_id
        self.device = device
        self.dir_dict = dir_dict

        self.criterion = criterion
        self.max_depth = max_depth

        log_path = "%s/task_%d.txt" % (dir_dict['log_dir'], exp_id)
        self.logger = get_logger('task_%d' % exp_id, log_path, output=False)

        self.pic_dir = dir_dict['pic_dir']
        self.mod_dir = dir_dict['mod_dir']

        super(NeuralDecisionTree, self).__init__()
        self.use_cuda = False
        self.dtype = torch.FloatTensor

        self.head_node = None

        self.depth = 0
        self.node_num = 0
        self.leaf_num = 0
        self.class_label = None
        self.feature_importance = None
        self.best_epoch = None

        self.n_class = None

        self.head_node = NeuralDecisionNode(node_id="h", device=self.device, criterion=self.criterion,
                                            use_cuda=self.use_cuda)

        self.need_visual = need_visual
        self.random_state = random_state

    def cuda(self):
        self.use_cuda = True
        self.dtype = torch.cuda.FloatTensor
        super(NeuralDecisionTree, self).to(self.device)
        self.head_node.foreach(lambda node: node.cuda())

    def fit(self, trn_set):

        self.class_label = trn_set.class_label
        self.n_class = trn_set.n_class
        self.head_node.n_class = trn_set.n_class

        self.grow(self.head_node, trn_set, depth=self.depth)

    def grow(self, node, trn_set, depth):

        if depth > self.depth:
            self.depth = depth
        self.node_num += 1

        node_y = trn_set.targets

        node.depth = depth
        node.sample_num = len(node_y)
        one_hot_target = torch.nn.functional.one_hot(node_y, num_classes=self.n_class).float()
        node.class_dist = one_hot_target.sum(dim=0).long()
        node.prob_dist = one_hot_target.mean(dim=0)
        node.class_prob = one_hot_target.mean(dim=0)
        node.impurity = calc_impurity(node_y, self.criterion)

        if self.is_leaf(node, node_y):
            node.is_leaf = True
            self.leaf_num += 1
            output_node_info(self.logger, node)
            return
        else:
            node.is_leaf = False
        del node_y

        node.init_split(trn_set.features.shape)

        node.refine(trn_set)

        left_idx, right_idx = node.split(trn_set)

        if len(left_idx) == 0 or len(right_idx) == 0:
            node.is_leaf = True
            node.reason = "Failed split!"
            self.leaf_num += 1

            output_node_info(self.logger, node)
        else:
            node.is_leaf = False

            output_node_info(self.logger, node)

            _, left_x, left_y = trn_set[left_idx]
            _, right_x, right_y = trn_set[right_idx]
            left_train_set = package_data(left_x, left_y, left_y.dtype, self.device)
            right_train_set = package_data(right_x, right_y, left_y.dtype, self.device)

            self.grow(node.left_child, left_train_set, depth + 1)
            self.grow(node.right_child, right_train_set, depth + 1)

    def forward(self, features, sup_set=None):

        sup_prob = None
        if sup_set is not None:
            sup_prob = torch.tensor([1. for _ in range(len(sup_set.targets))], dtype=torch.float32)
            if self.use_cuda:
                sup_prob = sup_prob.to(self.device)
        return self.head_node(features, sup_set, sup_prob)

    def predict(self, features):
        proba = self.predict_proba(features)
        return np.argmax(proba, axis = 1)

    def predict_proba(self, features, targets=None, name='default'):
        self.eval()
        if targets is not None:
            sample_prob = torch.tensor(1.).repeat(features.shape[0])
            if self.use_cuda:
                sample_prob = sample_prob.to(self.device)
            # save_path = f'{self.pic_dir}/decision_{self.exp_id}_{name}_soft'
            # visualize_decision(self.head_node, save_path, features, targets, sample_prob)
        return self.head_node(features).detach().cpu().numpy()

    def is_leaf(self, node, targets):
        # 判断数据是否只有一类
        if len(torch.unique(targets)) == 1:
            node.reason = 'one_class'
            return True
        # 判断是否达到最大深度
        if self.max_depth is not None and node.depth == self.max_depth:
            node.reason = 'max_depth'
            return True
        return False

    def refine(self, data_set):

        self.train()
        self.change_param_state(self.head_node, 'layer1', False)
        self.change_param_state(self.head_node, 'layer2', True)

        train_set, valid_set = data_set['Trn'], data_set['Vld']

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=0.0)

        max_epoch = 1000
        r = 0.5

        scheduler = CosineAnnealingLR(optimizer, T_max=max_epoch)

        trn_loss_s, trn_f1_s, vld_loss_s, vld_f1_s, loss_s = [], [], [], [], []

        pbar = tqdm(range(1, max_epoch + 1))
        for epoch in pbar:

            trn_output = self.forward(train_set.features, train_set)
            trn_loss = criterion(trn_output, train_set.targets)
            vld_output = self.forward(valid_set.features)
            vld_loss = criterion(vld_output, valid_set.targets)

            loss = trn_loss * r + vld_loss * (1 - r)

            l2_reg = torch.tensor(0., dtype=torch.float32).to(self.device)
            for name, param in self.named_parameters():
                if "layer2.weight" in name:
                    l2_reg += torch.sum(torch.square(param))
            loss = loss + 0.01 * l2_reg

            with torch.no_grad():
                trn_loss, trn_f1 = calc_perf(self, train_set.features, train_set.targets, criterion)
                vld_loss, vld_f1 = calc_perf(self, valid_set.features, valid_set.targets, criterion)

                trn_loss_s.append(trn_loss.item())
                trn_f1_s.append(trn_f1)
                vld_loss_s.append(vld_loss.item())
                vld_f1_s.append(vld_f1)
                loss_s.append(loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            scheduler.step()

            show_dict = {
                'TrL': trn_loss.item(),
                'TrF': trn_f1,
                'VaL': vld_loss.item(),
                'VaF': vld_f1,
                'Loss': loss.item(),
                'LR': optimizer.param_groups[0]['lr']
            }
            format_str = "{:.6f}"
            pbar.set_postfix({key: format_str.format(value) for key, value in show_dict.items()})

        if self.need_visual:
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
            indices = np.arange(0, len(trn_loss_s), 1)
            ax1.plot(indices, trn_loss_s, color='#c1f1fc', label='trn_loss')
            ax1.plot(indices, vld_loss_s, color='#ebffac', label='vld_loss')
            ax1.plot(indices, loss_s, color='#808080', label='loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.tick_params(axis='y')
            ax2 = ax1.twinx()
            ax2.plot(indices, trn_f1_s, color='#00c7f2', label='trn_f1')
            ax2.plot(indices, vld_f1_s, color='#c2ff00', label='vld_f1')
            ax2.set_ylabel('F1')
            ax2.tick_params(axis='y')
            max_i = np.argmax(np.array(trn_f1_s))
            ax2.text(indices[max_i], trn_f1_s[max_i], "(%d, %.4f)" % (max_i + 1, trn_f1_s[max_i]),
                     ha='center', fontsize=10)
            max_i = np.argmax(np.array(vld_f1_s))
            ax2.text(indices[max_i], vld_f1_s[max_i], "(%d, %.4f)" % (max_i + 1, vld_f1_s[max_i]),
                     ha='center', fontsize=10)
            min_i = np.argmin(np.array(loss_s))
            ax1.text(indices[min_i], loss_s[min_i], "(%d, %.4f)" % (min_i + 1, loss_s[min_i]),
                     ha='center', fontsize=10)
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2)
            plt.title('Loss & F1 for epoch')
            plt.grid(True)
            save_name = "{}/Tuning_curve_{}.png".format(self.pic_dir, self.exp_id)
            plt.savefig(save_name, dpi=300, bbox_inches='tight')
            plt.close()

        torch.save(self.state_dict(), "{}/Param_{}.pth".format(self.mod_dir, self.exp_id))

        return

    def change_param_state(self, node, key, update):
        if not node.is_leaf:
            for name, param in node.named_parameters():
                if key in name:
                    if update:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            self.change_param_state(node.left_child, key, update)
            self.change_param_state(node.right_child, key, update)

    def get_feature_importance(self):
        self.feature_importance = get_node_fi(self.head_node).detach().cpu().numpy()
        return self.feature_importance


def get_node_fi(node):
    total_fi = node.path_prob * node.gate.layer2.weight.data[0]
    if not node.left_child.is_leaf:
        total_fi += get_node_fi(node.left_child)
    if not node.right_child.is_leaf:
        total_fi += get_node_fi(node.right_child)
    return total_fi




