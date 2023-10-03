########################################################################
#                       Linear            Models                       #
########################################################################
import math
import os
import pickle
import numpy as np
import torch
from hyperopt import hp
from sklearn.linear_model import Lasso
from sklearn.preprocessing import Normalizer


from RDBLab.utils.metric import Metric

# For computing loss
def squared_diff(output, target):
    return torch.sum((output - target) ** 2)


class Linear_Regress():
    def __init__(self, opt, pass_dim_dict):
        self.device = torch.device('cuda:0') if torch.cuda.is_available() \
            else torch.device('cpu:0')
        self.save_dir = opt.mid_data_dir + opt.save_dir + "/" + str(opt.batch_size)
        self.test = False
        self.eval = False
        # self.test_time = opt.test_time
        self.batch_size = opt.batch_size
        self.dataset = opt.dataset
        self.latest_save_freq = opt.save_latest_epoch_freq

        self.dim_dict = pass_dim_dict

        self.hyperspace = {
            'batch_size': hp.choice('batch_size', [2 ** i for i in range(10)]),  # 获取2的幂次方作为batch测试
            'alpha': hp.randint('alpha', 5),
            'max_iter': hp.choice('max_iter', [100000, 200000, 300000, 500000, 700000, 1000000]),
            'tol': hp.choice('tol', [1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 10])
        }
        self.best_total_loss = math.inf
        self.best_epoch = 0
        self.test_loss = None
        self.last_total_loss = 100000
        self.last_test_loss = None
        self.last_pred_err = None
        self.last_mse_err = None
        self.pred_err = None
        self.mse_err = None
        self.rq = 0
        self.last_rq = 0

        if not os.path.exists(opt.mid_data_dir + opt.save_dir):
            os.mkdir(opt.mid_data_dir + opt.save_dir)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # Initialize the neural units
        self.units = {}
        self.models = {}
        self.best = 100000

        for operator in self.dim_dict:
            self.models[operator] = Lasso(alpha=10, warm_start=True, max_iter=1000000, tol=1e-01)

        self.loss_fn = squared_diff
        # Initialize the global loss accumulator dict
        self.dummy = torch.zeros(1).to(self.device)
        self.acc_loss = {operator: [] for operator in self.dim_dict}
        self.curr_losses = {operator: 0 for operator in self.dim_dict}
        self.total_loss = None
        self._test_losses = dict()

        if opt.mode=='test':
            self.load()

    def set_input(self, samp_dicts):
        self.input = samp_dicts

    def number_to_output_vec(self, number, dimx, dimy):
        # 创建一个dimx行dimy维的零矩阵
        vector = np.zeros((dimx, dimy))
        # 将每一行的首元素设置为给定的数值
        vector[:, 0] = number
        return vector

    def regress_OneElement(self, samp_batch):
        feat_vec = samp_batch['feat_vec']
        y_time = samp_batch['total_time']
        y_time = y_time.reshape(-1, 1)

        last_model_score = 0

        scaler = Normalizer()
        input_vec = scaler.fit_transform(feat_vec)
        subplans_time = []
        for child_plan_dict in samp_batch['children_plan']:
            child_output_vec, _ = self.regress_OneElement(child_plan_dict)
            if not child_plan_dict['is_subplan']:
                input_vec = np.concatenate((input_vec, child_output_vec), axis=1)
            else:
                # print(child_output_vec.shape[0])
                # print('shape ---------------------------------')
                subplans_time.append(child_output_vec[:, 0].reshape(child_output_vec.shape[0], 1))
        expected_len = self.dim_dict[samp_batch['node_type']]
        # print(expected_len)
        # time.sleep(0.5)
        if expected_len > input_vec.shape[1]:
            add_on = np.zeros([input_vec.shape[0], expected_len - input_vec.shape[
                1]])  # input_vec.shape[0]是有几个算子，expected_len - input_vec.shape[1]是算子缺多少维
            input_vec = np.concatenate((input_vec, add_on), axis=1)

        y_time_raveled = y_time.ravel()
        if not self.test:
            self.models[samp_batch['node_type']].fit(input_vec, y_time_raveled)

        test_x = np.mean(input_vec, axis=0)  # 粗略的取平均值试试看。。。。
        output_vec = []
        pred_time_dict = []
        y_pred = self.models[samp_batch['node_type']].predict(input_vec)

        for vec in input_vec:
            pred_vec = self.models[samp_batch['node_type']].predict([vec])
            try:
                pred_time_dict.append([pred_vec[0][0]])
            except IndexError:
                pred_vec = [pred_vec]
                pred_time_dict.append([pred_vec[0][0]])
            output_vec.extend(pred_vec)

        output_vec = np.array(output_vec)
        pred_time_dict = np.array(pred_time_dict)
        cat_res = np.concatenate([pred_time_dict] + subplans_time, axis=1)
        pred_time = np.sum(cat_res, 1)
        loss = (pred_time - samp_batch['total_time']) ** 2
        # print(loss)
        # print("loss.shape", loss.shape)
        self.acc_loss[samp_batch['node_type']].append(loss)
        return output_vec, pred_time

    def regress(self, epoch):
        data_size = 0
        total_loss = 0
        total_losses = {operator: [] \
                        for operator in self.dim_dict}
        if self.test:
            test_loss = []
            pred_err = []
            mse_err = []

        if self.eval:
            self.total_times = []
            self.pred_times = []

        all_tt, all_pred_time = None, None

        total_mean_mae = 0
        for idx, samp_dict in enumerate(self.input):
            del self.acc_loss
            self.acc_loss = {operator: [] for operator in self.dim_dict}

            _, pred_time = self.regress_OneElement(samp_dict)
            if self.dataset == "PSQLTPCH":
                epsilon = 1.1920928955078125e-07
            else:
                epsilon = 0.001
            data_size += len(samp_dict['total_time'])
            if self.test:
                tt = samp_dict['total_time']

                if self.eval:
                    self.total_times.append(tt)
                    self.pred_times.append(pred_time)

                test_loss.append(np.abs(tt - pred_time))
                curr_pred_err = Metric.pred_err_numpy(tt, pred_time, epsilon)
                curr_mse_err = Metric.mse_numpy(tt, pred_time, epsilon)
                pred_err.append(curr_pred_err)
                mse_err.append(curr_mse_err)

                all_tt = tt if all_tt is None else np.concatenate((all_tt, tt), axis=0)
                all_pred_time = pred_time if all_pred_time is None \
                    else np.concatenate((all_pred_time, pred_time), axis=0)

                curr_rq = Metric.r_q_numpy(tt, pred_time, epsilon)

                curr_mean_mae = Metric.mean_mae_numpy(tt, pred_time, epsilon)
                total_mean_mae += curr_mean_mae * len(tt)

                if epoch % 50 == 0:
                    print("####### eval by temp: idx {}, test_loss {}, pred_err {},  mse_err {}, " \
                          "rq {}, weighted mae {}, accumulate_err {} " \
                          .format(idx, np.mean(np.abs(tt - pred_time)),
                                  np.mean(curr_pred_err),np.sqrt(np.mean(curr_mse_err)),
                                  curr_rq, curr_mean_mae,
                                  Metric.accumulate_err_numpy(tt, pred_time, epsilon)))
            D_size = 0
            subbatch_loss = 0
            for operator in self.acc_loss:
                # print(operator, self.acc_loss[operator])
                # time.sleep(5)
                all_loss = np.array([])
                all_loss = np.append(all_loss, self.acc_loss[operator])
                D_size += all_loss.shape[0]
                subbatch_loss += np.sum(all_loss)

                total_losses[operator].append(all_loss)

            subbatch_loss = np.mean(np.sqrt(subbatch_loss / D_size))
            # print("subbatch_loss.shape",subbatch_loss.shape)
            total_loss += subbatch_loss * samp_dict['subbatch_size']

        if self.test:
            all_test_loss = np.concatenate(test_loss, axis=0)

            all_test_loss = np.mean(all_test_loss)
            self.test_loss = all_test_loss

            all_mse_err = np.concatenate(mse_err)
            self.mse_err = np.sqrt(np.mean(all_mse_err))

            all_pred_err = np.concatenate(pred_err, axis=0)
            self.pred_err = np.mean(all_pred_err)

            self.rq = Metric.r_q_numpy(all_tt, all_pred_time, epsilon)
            self.accumulate_err = Metric.accumulate_err_numpy(all_tt, all_pred_time,
                                                              epsilon)
            self.weighted_mae = total_mean_mae / data_size

            if epoch % 50 == 0:
                print("test batch Pred Err: {}, Mse Err: {}, R(q): {}, Accumulated Error: " \
                      "{}, Weighted MAE: {}".format(self.pred_err,
                                                    self.mse_err,
                                                    self.rq,
                                                    self.accumulate_err,
                                                    self.weighted_mae))
        else:
            self.total_loss = np.mean(total_loss / self.batch_size)
            self.curr_losses = {operator: np.mean(np.concatenate(total_losses[operator])) for operator in
                                self.dim_dict}


    def get_current_losses(self):
        return self.curr_losses

    def evaluate(self, dataset):
        self.test = True
        self.set_input(dataset)
        self.eval = True
        self.regress(0)
        self.last_total_loss = self.total_loss
        self.last_test_loss = self.test_loss
        self.last_pred_err = self.pred_err
        self.last_mse_err = self.mse_err
        self.last_rq = self.rq
        self.test_loss, self.pred_err,self.mse_err = None, None, None
        self.rq = 0

        return self.pred_times, self.total_times

    def load(self):
        paths = os.listdir(self.save_dir)
        for model_name in paths:
            model_name_without_extension = model_name.split("_")[0]
            with open(os.path.join(self.save_dir, model_name), 'rb') as f:
                self.models[model_name_without_extension] = pickle.load(f)

    def optimize_parameters(self, epoch):
        self.test = False
        self.regress(epoch)

        self.input = self.test_dataset
        self.test = True
        self.regress(epoch)
        self.save_units(epoch)
        self.last_test_loss = self.test_loss
        self.last_pred_err = self.pred_err
        self.last_mse_err = self.mse_err
        self.last_rq = self.rq
        self.test_loss, self.pred_err,self.mse_err = None, None, None
        self.rq = 0

    def save_units(self, epoch):
        if self.total_loss < self.best_total_loss:
            self.best_total_loss = self.pred_err
            self.best_epoch = epoch

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            for operator in self.dim_dict:
                model_path = os.path.join(self.save_dir, operator + '_best.pickle')
                with open(model_path, 'wb') as file_model:
                    pickle.dump(self.models[operator], file_model)
