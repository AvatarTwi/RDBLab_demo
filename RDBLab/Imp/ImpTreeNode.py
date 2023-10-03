import numpy as np
import torch


class ImpTreeNode:
    def __init__(self, root):
        self.root = root

        self.reference_len = 100
        self.data_len = 100
        # np.array 2
        self.feat_vec = None
        # list(np.array 2)
        self.child_vec = []
        # np.array 1
        self.true_time = None
        # np.array 1=self.output[:,0]
        self.pred_time = None
        # np.array 2
        self.output = None
        # np.array 1
        self.importance_feat = None
        # list(np.array 1)
        self.importance_child = []
        # np.array 1: 由父节点传递
        self.importance_output = None
        # list(ImpTreeNode)
        self.childs = []
        # DataStore
        self.background = None
        # DataStore
        self.data = None

    # 迭代计算重要性的主函数
    def count_importance(self):
        self.split_background_data()
        self.cal_importance_feat()
        self.cal_importance_child()
        print(self.importance_output)
        print(self.importance_feat)
        print(self.importance_child)
        for idx, child in enumerate(self.childs):
            child.importance_output = self.importance_child[idx]
            child.count_importance()

    def split_background_data(self):
        bg = np.random.choice(np.arange(len(self.true_time)),
                              min(self.reference_len, int(len(self.true_time) * 0.8)), replace=False)

        dt = [i for i in np.arange(len(self.true_time)) if i not in bg]

        cvs = []
        for cv in self.child_vec:
            cvs.append(cv[bg])

        self.background = DataStore(self.feat_vec[bg], cvs,
                                    self.true_time[bg], self.pred_time[bg], self.output[bg],
                                    self.importance_output)

        cvs = []
        for cv in self.child_vec:
            cvs.append(cv[dt])

        self.data = DataStore(self.feat_vec[dt], cvs,
                              self.true_time[dt], self.pred_time[dt], self.output[dt],
                              self.importance_output)

    # 计算feat_vec的重要性
    def cal_importance_feat(self):
        if self.root:
            delta_ys = [torch.abs(self.background.true_time - tt) for tt in self.data.true_time]
            delta_xs = [torch.abs(self.background.feat_vec - fv) for fv in self.data.feat_vec]
        else:
            output = self.data.output * self.importance_output
            delta_ys = [torch.abs(self.background.output * self.importance_output - op) for op in output]
            delta_ys = [torch.sum(delta_y) for delta_y in delta_ys]
            delta_xs = [torch.abs(self.background.feat_vec - fv) for fv in self.data.feat_vec]

        self.importance_feat = []
        for i, delta_x in enumerate(delta_xs):
            # 每个data中的点和background中的点的delta
            # delta为list, 长度为feat_vec的长度，体现的是feat_vec中每个特征的重要性
            delta = [torch.div(delta_ys[i][j], each) for j, each in enumerate(delta_x)]
            if not self.importance_feat:
                self.importance_feat = delta
            else:
                self.importance_feat += delta

    # 计算child_vec的重要性
    def cal_importance_child(self):
        for idx,_ in enumerate(self.data.child_vec):
            if self.root:
                delta_ys = [torch.abs(self.background.true_time - tt) for tt in self.data.true_time]
                delta_xs = [torch.abs(self.background.child_vec[idx] - fv) for fv in self.data.child_vec[idx]]
            else:
                output = self.data.output * self.importance_output
                delta_ys = [torch.abs(self.background.output * self.importance_output - op) for op in output]
                delta_ys = [torch.sum(delta_y) for delta_y in delta_ys]
                delta_xs = [torch.abs(self.background.child_vec[idx] - fv) for fv in self.data.child_vec[idx]]

            self.importance_child = []
            for i, delta_x in enumerate(delta_xs):
                # 每个data中的点和background中的点的delta
                # delta为list, 长度为feat_vec的长度，体现的是feat_vec中每个特征的重要性
                delta = [torch.div(delta_ys[i][j], each) for j, each in enumerate(delta_x)]
                if not self.importance_child:
                    self.importance_child = delta
                else:
                    self.importance_child += delta


class DataStore:
    def __init__(self, feat_vec, child_vec, true_time, pred_time, output, importance_output):
        # np.array 2
        self.feat_vec = feat_vec
        # list(np.array 2)
        self.child_vec = child_vec
        # np.array 1
        self.true_time = true_time
        # np.array 1=self.output[:,0]
        self.pred_time = pred_time
        # np.array 2
        self.output = output
        # np.array 1: 由父节点传递
        self.importance_output = importance_output
