import os
import pickle

from matplotlib import pyplot as plt
from matplotlib.pyplot import *

# 设置字体大小

from RDBLab.utils.metric import Metric

rc('font', size=15)


def plot_bar(type, dict_all):
    # 设置画布大小
    fig = plt.figure(figsize=(8 * len(dict_all.keys()), 7))

    fig.subplots_adjust(left=0.035, right=0.980, top=0.9, bottom=0.20,
                        wspace=0.2, hspace=0.3)

    count = 1

    for i in range(len(type)):

        try:
            ax = fig.add_subplot(1, 6, count)
            count += 1

            temp_error = []
            x_label = []

            for key in dict_all.keys():
                x_label.append(key)
                temp_error.append(dict_all[key][type[i]])

            colors = [sum([ord(i) * 10 for i in idx.split("_")[0]]) for idx in x_label]
            colors = ["#" + str(c) + "FF" for c in colors]

            barlist = ax.bar(x_label, temp_error, color=colors)
            for idx, l in enumerate(x_label):
                if 'origin' in l:
                    barlist[idx].set_color("r")

            for a, b in enumerate(temp_error):  # 柱子上的数字显示
                plt.text(a, b, '%.3f' % b, ha='center', va='bottom', fontsize=6)

            plt.title(type[i])
            plt.xticks(x_label, rotation=45, fontsize=12)
        except:
            continue

    return plt


def plot_scatter(root_dir, model_type, dict_all):
    # 设置画布大小
    fig = plt.figure(figsize=(8 * len(dict_all.keys()), 7))

    fig.subplots_adjust(left=0.02, right=0.920, top=0.95, bottom=0.05,
                        wspace=0.3, hspace=0.3)

    fig.tight_layout()

    count = 1
    for key in dict_all.keys():
        ax = fig.add_subplot(1, len(dict_all.keys()), count)
        count += 1

        max_pred = max(dict_all[key]['pred_times']) * 1.1
        max_tt = max(dict_all[key]['total_times']) * 1.1
        ax.plot(
            [0, max(max_tt, max_pred)],
            [0, max(max_tt, max_pred)],
            linewidth=1,
            color="black",
            linestyle="--",
        )
        ax.scatter(
            dict_all[key]['total_times'],
            dict_all[key]['pred_times'],
            marker="x",
            color="r",
            label="Optimized",
        )

        lim = max(max_tt, max_pred)
        lim_low = min(0, min(min(dict_all[key]['pred_times']),min(dict_all[key]['total_times'])))
        # ax.set_xscale('log', base=2)
        ax.set_xbound((lim_low, lim))
        # ax.set_yscale('log', base=2)
        ax.set_ybound((lim_low, lim))

        txt = []
        try:
            for k in dict_all[key]['values'].keys():
                txt.append(k + ":" + str('%.3f' % dict_all[key]['values'][k]))
        except:
            pass

        ax.text(
                x=max_tt * 0.7, y=max_pred / 20,
                s="\n".join(txt), fontsize=12
                )

        ax.set_title(key.replace("_exp1", "").replace("_5", "")
                     .replace("tpcc_", ""), fontsize=20, fontweight="bold")

    plt.savefig(root_dir + "/" + model_type + "_scatters.png")


def scatter(opt, model_type):
    dict_all = {}
    pattern_num = re.compile(r'\d+.?\d*\w')
    pattern_type = re.compile(r'\w*\(?\w*\)?:')
    root_dir = opt.version

    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            if "model" not in dir:
                continue
            save_model = "/" + dir + opt.save_dir + "/" + str(opt.batch_size)
            for root1, dirs1, files1 in os.walk(root_dir + save_model):
                print(root_dir + save_model)
                if [file for file in files1 if 'times' in file] is not None:
                    with open(root_dir + save_model + "/pred_times.pickle", "rb") as f:
                        pred_times = pickle.load(f)
                        if 'Net_v2' in opt.save_dir or 'QPPNet' in opt.save_dir:
                            pred_timess = [time.cpu().detach().numpy() for time in pred_times]
                        else:
                            pred_timess = [time for time in pred_times]
                    with open(root_dir + save_model + "/total_times.pickle", "rb") as f:
                        total_times = pickle.load(f)
                        if 'Net_v2' in opt.save_dir or 'QPPNet' in opt.save_dir:
                            total_timess = [time.cpu().detach().numpy() for time in total_times]
                        else:
                            total_timess = [time for time in total_times]
                else:
                    continue


                for idx, pt in enumerate(pred_timess):
                    tempTT = total_timess[idx]
                    tempPT = pt
                    if idx == 0:
                        pred_times = list(tempPT)
                        total_times = list(tempTT)
                        continue
                    pred_times.extend(list(tempPT))
                    total_times.extend(list(tempTT))

                print(len(total_times))

                dict_all[dir] = {}
                dict_all[dir]['pred_times'] = pred_times
                dict_all[dir]['total_times'] = total_times

                x_ = pred_times - np.mean(pred_times)
                y_ = total_times - np.mean(total_times)
                r = np.dot(x_, y_) / (np.linalg.norm(x_) * np.linalg.norm(y_))

                q_error = Metric.q_error_numpy(total_times, pred_times, 0.001)

                dict = {}
                dict['pearson'] = r
                dict['q_error'] = q_error[4]
                dict['q_error99'] = q_error[0]
                dict['q_error95'] = q_error[1]
                dict['q_error90'] = q_error[2]

                with open(root_dir + save_model + "/eval.txt", "r+") as f:
                    txt = f.read()
                    type = [i.replace(":", "") for i in pattern_type.findall(txt)]
                    num = [float(i.replace("s", "")) for i in pattern_num.findall(txt)]

                try:
                    for idx, t in enumerate(type):
                        dict[t] = num[idx]

                except:
                    pass
                dict_all[dir]['values'] = dict

    plot_scatter(root_dir, model_type, dict_all)


def bar(opt, model_type):
    pattern_num = re.compile(r'\d+.?\d*\w')
    pattern_type = re.compile(r'\w*\(?\w*\)?:')
    root_dir = opt.version

    dict_all = {}
    type = []

    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            if "model" not in dir:
                continue
            save_model = "/" + dir + opt.save_dir + "/" + str(opt.batch_size)
            for root1, dirs1, files1 in os.walk(root_dir + save_model):
                for file in files1:
                    if "eval.txt" not in file:
                        continue

                    dict = {}
                    with open(root_dir + save_model + "/" + file, "r+") as f:
                        txt = f.read()
                        num = [float(i.replace("s", "")) for i in pattern_num.findall(txt)]
                    try:
                        for idx, t in enumerate(type):
                            dict[t] = num[idx]
                        dir = dir.replace("_exp1", "").replace("_5", "").replace("tpcc_", "")
                        dict_all[dir] = dict
                    except:
                        pass

    print(dict_all)
    plt = plot_bar(type, dict_all)
    plt.savefig(root_dir + "/" + model_type + "_compare.png")
