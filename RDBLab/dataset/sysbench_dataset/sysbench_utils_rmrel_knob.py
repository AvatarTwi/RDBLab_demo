import collections
import json
import os
import pickle
import random
import re
import shutil

import numpy as np
from sklearn.model_selection import train_test_split

from RDBLab.dataset.sysbench_dataset.attr_rel_dict import *
from RDBLab.dataset.sysbench_dataset.cost_factor.cost_factor import cost_factor_main, cost_factor_one2one
from RDBLab.dataset.sysbench_dataset.attr_rel_dict import aggreg_strats
import RDBLab.config
basics = 3  # get_basics(plan_dict)
SCALE = 1

div = 90
num_per_q = 120 * 90 * 7 * 7 * 7 / div
knob_num = 3
op_data_dic = {}

TRAIN_TEST_SPLIT = 0.8


def sysbench_dim(dic_factor):
    sysbench_dim_dict = {'Seq Scan': knob_num + basics + len(dic_factor["Seq Scan"]),
                     'Index Scan': knob_num + basics + 1 + len(dic_factor["Index Scan"]),
                     'Index Only Scan': knob_num + basics + 1 + len(dic_factor["Index Only Scan"]),
                     'Bitmap Heap Scan': knob_num + basics + 32 + len(dic_factor["Bitmap Heap Scan"]),
                     'Bitmap Index Scan': knob_num + 3 + len(dic_factor["Bitmap Index Scan"]),
                     'Sort': knob_num + basics + len(sort_algos) + 32 + len(dic_factor["Sort"]),
                     'Hash': knob_num + basics + 1 + 32 + len(dic_factor["Hash"]),
                     'Hash Join': knob_num + basics + len(join_types) + len(parent_rel_types) + 32 * 2 + len(
                         dic_factor["Hash Join"]),
                     'Merge Join': knob_num + basics + len(join_types) + len(parent_rel_types) + 32 * 2 + len(
                         dic_factor["Merge Join"]),
                     'Aggregate': knob_num + basics + len(aggreg_strats) + 1 + 32 + len(dic_factor["Aggregate"]),
                     'Nested Loop': knob_num + 32 * 2 + basics + len(dic_factor["Nested Loop"]),
                     'Limit': knob_num + 32 + basics + len(dic_factor["Limit"]),
                     'Subquery Scan': knob_num + 32 + basics + len(dic_factor["Subquery Scan"]),
                     'Materialize': knob_num + 32 + basics + len(dic_factor["Materialize"]),
                     'Gather Merge': knob_num + 32 + basics + len(dic_factor["Gather Merge"]),
                     'Gather': knob_num + 32 + basics + len(dic_factor["Gather"]),
                     'BitmapAnd': knob_num + 32 * 2 + basics + len(dic_factor["BitmapAnd"]),
                     'Memoize': knob_num + 32 + basics + len(dic_factor["Memoize"]),
                     'ModifyTable':  num_rel + 32 + basics + 2,
                     'Result': basics + 2,
                     'LockRows': 32 + basics + 2,
                     'Append': 32 + basics + 2,
                     'Unique': 32 + basics + 2,
                     }
    return sysbench_dim_dict


# need to normalize Plan Width, Plan Rows, Total Cost, Hash Bucket
def get_basics(plan_dict):
    return [plan_dict['Plan Width'], plan_dict['Plan Rows'], plan_dict['Total Cost']]


def get_scan_input(plan_dict):
    return get_basics(plan_dict)


def get_index_scan_input(plan_dict):
    res = get_basics(plan_dict) + [1 if plan_dict['Scan Direction'] == 'Forward' else 0]

    return res


def get_bitmap_index_scan_input(plan_dict):
    # plan_dict: dict where the plan_dict['node_type'] = 'Bitmap Index Scan'

    return get_basics(plan_dict)


def get_hash_input(plan_dict):
    return get_basics(plan_dict) + [plan_dict['Hash Buckets'] if 'Hash Buckets' in plan_dict.keys() else 1]


def get_join_input(plan_dict):
    type_vec = [0] * len(join_types)
    type_vec[join_types.index(plan_dict['Join Type'].lower())] = 1
    par_rel_vec = [0] * len(parent_rel_types)
    if 'Parent Relationship' in plan_dict:
        par_rel_vec[parent_rel_types.index(plan_dict['Parent Relationship'].lower())] = 1
    return get_basics(plan_dict) + type_vec + par_rel_vec

def get_nested_input(plan_dict):
    type_vec = [0] * len(join_types)
    type_vec[join_types.index(plan_dict['Join Type'].lower())] = 1
    
    return get_basics(plan_dict) + type_vec


def get_sort_input(plan_dict):
    sort_meth = [0] * len(sort_algos)
    if 'Sort Method' in plan_dict:
        if "external" not in plan_dict['Sort Method'].lower():
            sort_meth[sort_algos.index(plan_dict['Sort Method'].lower())] = 1

    return get_basics(plan_dict) + sort_meth


def get_aggreg_input(plan_dict):
    strat_vec = [0] * len(aggreg_strats)
    strat_vec[aggreg_strats.index(plan_dict['Strategy'].lower())] = 1
    partial_mode_vec = [0] if plan_dict['Parallel Aware'] == 'false' else [1]
    return get_basics(plan_dict) + strat_vec + partial_mode_vec


def get_modify_input(plan_dict):
    # plan_dict: dict where the plan_dict['node_type'] = 'Seq Scan'
    rel_vec = get_rel_one_hot(plan_dict['Relation Name'])

    return get_basics(plan_dict) + rel_vec


sysbench_GET_INPUT = \
    {
        "Hash Join": get_join_input,
        "Merge Join": get_join_input,
        "Nested Loop": get_nested_input,
        "Seq Scan": get_scan_input,
        "Index Scan": get_index_scan_input,
        "Index Only Scan": get_index_scan_input,
        "Bitmap Heap Scan": get_scan_input,
        "Bitmap Index Scan": get_bitmap_index_scan_input,
        "Sort": get_sort_input,
        "Hash": get_hash_input,
        "Aggregate": get_aggreg_input,
        "ModifyTable": get_modify_input,
    }

sysbench_GET_INPUT = collections.defaultdict(lambda: get_basics, sysbench_GET_INPUT)


sysbench_OP = ['ModifyTable', 'Result', 'LockRows']

def verify(data):
    if data['Node Type'] in sysbench_OP:
        return False

    if 'Plans' in data:
        for sub_tree in data['Plans']:
            return verify(sub_tree)

    return True
###############################################################################
#       Parsing data from RDBLab.csv files that contain json output of queries       #
###############################################################################

class SysbenchDataset():
    def __init__(self, opt):
        """
            Initialize the dataset by parsing the data files.
            Perform train test split and normalize each feature using mean and max of the train dataset.

            self.dataset is the train dataset
            self.test_dataset is the test dataset
        """

        mid_data_dir = opt.mid_data_dir
        self.num_sample_per_q = int(RDBLab.config.num_per_q[-1]*TRAIN_TEST_SPLIT)

        if not os.path.exists(mid_data_dir):
            os.makedirs(mid_data_dir)

        self.batch_size = opt.batch_size
        self.num_q = 1
        self.SCALE = SCALE
        self.input_func = sysbench_GET_INPUT
        self.cost_factor_dict = {}

        if opt.new_mid_data:

            self.input_func = sysbench_GET_INPUT

            self.grp_idxes = []
            self.num_grps = [0] * self.num_q

            data = []
            datas = {}
            all_groups, all_groups_test, all_groups_train = [], [],[]

            if opt.new_data_structure:
                for root, dirs, files in os.walk(opt.data_dir):
                    for dir in dirs:
                        flag = 0
                        for i in dir.split("-"):
                            if int(i) % 200 == 0:
                                flag = 1
                                break
                        if flag == 1:
                            continue
                        temp_data = []
                        for i in range(self.num_q):
                            fname = root + "/" + dir + "/serverlog"
                            data1 = self.get_all_plans(fname)
                            temp_data.append(data1[:int(len(data1)/div)])

                        temp_data = cost_factor_one2one(opt,dir, temp_data)

                        for i in range(self.num_q):
                            if i not in datas.keys():
                                datas[i] = []
                            datas[i].extend(temp_data[i])
                with open(opt.data_structure + '/datas.pickle', 'wb') as f:
                    pickle.dump(datas, f)
            else:
                with open(opt.data_structure + '/datas.pickle', 'rb') as f:
                    datas = pickle.load(f)

            for i in datas.keys():
                temp_data = datas[i]

                ##### this is for all samples for this query template #####
                enum, num_grp = self.grouping(temp_data)
                groups = [[] for _ in range(num_grp)]
                for j, grp_idx in enumerate(enum):
                    groups[grp_idx].append(temp_data[j])
                all_groups += groups

                ##### this is for train #####
                TrainEnum, testEnum, TrainTempdata, testTempdata = \
                    train_test_split(enum, temp_data, train_size=int(self.num_sample_per_q),test_size=int(self.num_sample_per_q/4), random_state=1)
                self.grp_idxes += TrainEnum
                self.num_grps[i] = num_grp
                data += TrainTempdata

                ##### this is for test #####
                test_groups = [[] for _ in range(num_grp)]
                for j, grp_idx in enumerate(testEnum):
                    test_groups[grp_idx].append(testTempdata[j])
                all_groups_test += test_groups

            self.dataset = data
            self.datasize = len(self.dataset)

            print("Number of groups per query: ", self.num_grps)

            if not os.path.exists(mid_data_dir):
                os.mkdir(mid_data_dir)

            self.mean_range_dict = self.normalize(all_groups_train)
            with open(mid_data_dir + '/mean_range_dict.pickle', 'wb') as f:
                pickle.dump(self.mean_range_dict, f)

            self.test_dataset = [self.get_input(grp) for grp in all_groups_test if grp != []]
            self.train_dataset = [self.get_input(grp) for grp in all_groups_train if grp != []]
            self.all_dataset = [self.get_input(grp) for grp in all_groups if grp != []]

            with open(mid_data_dir + '/grp_idxes.pickle', 'wb') as f:
                pickle.dump(self.grp_idxes, f)
            with open(mid_data_dir + '/num_grps.pickle', 'wb') as f:
                pickle.dump(self.num_grps, f)
            with open(mid_data_dir + '/data.pickle', 'wb') as f:
                pickle.dump(self.dataset, f)
            with open(mid_data_dir + '/test_dataset.pickle', 'wb') as f:
                pickle.dump(self.test_dataset, f)
            with open(mid_data_dir + '/train_dataset.pickle', 'wb') as f:
                pickle.dump(self.train_dataset, f)
            with open(mid_data_dir + '/all_dataset.pickle', 'wb') as f:
                pickle.dump(self.all_dataset, f)
        else:
            with open(mid_data_dir + '/grp_idxes.pickle', 'rb') as f:
                self.grp_idxes = pickle.load(f)
            with open(mid_data_dir + '/num_grps.pickle', 'rb') as f:
                self.num_grps = pickle.load(f)
            with open(mid_data_dir + '/data.pickle', 'rb') as f:
                self.dataset = pickle.load(f)
                self.datasize = len(self.dataset)
            with open(mid_data_dir + '/test_dataset.pickle', 'rb') as f:
                self.test_dataset = pickle.load(f)
            with open(mid_data_dir + '/train_dataset.pickle', 'rb') as f:
                self.train_dataset = pickle.load(f)
            with open(mid_data_dir + '/all_dataset.pickle', 'rb') as f:
                self.all_dataset = pickle.load(f)
            with open(mid_data_dir + '/mean_range_dict.pickle', 'rb') as f:
                self.mean_range_dict = pickle.load(f)

    def normalize(self):  # compute the mean and std vec of each operator
        """
            For each operator, normalize each input feature to have a mean of 0 and maximum of 1

            Returns:
            - mean_range_dict: a dictionary where the keys are the Operator Names and the values are 2-tuples (mean_vec, max_vec):
                -- mean_vec : a vector of mean values for input features of this operator
                -- max_vec  : a vector of max values for input features of this operator
        """
        feat_vec_col = {operator: [] for operator in all_dicts}

        def parse_input(data):
            # feat_vec = [self.input_func[data[0]["Node Type"]](jss) for jss in data]
            feat_vec = [np.hstack(
                (self.input_func[jss["Node Type"]](jss), jss['inner_vector']))
                for jss
                in data]

            if 'Plans' in data[0]:
                for i in range(len(data[0]['Plans'])):
                    parse_input([jss['Plans'][i] for jss in data])
            feat_vec_col[data[0]["Node Type"]].append(np.array(feat_vec).astype(np.float32))

        for i in range(self.datasize // self.num_sample_per_q):
            try:
                if self.num_grps[i] == 1:
                    parse_input(self.dataset[i * self.num_sample_per_q:(i + 1) * self.num_sample_per_q])
                else:
                    groups = [[] for j in range(self.num_grps[i])]
                    offset = i * self.num_sample_per_q
                    for j, plan_dict in enumerate(self.dataset[offset:offset + self.num_sample_per_q]):
                        groups[self.grp_idxes[offset + j]].append(plan_dict)
                    for grp in groups:
                        parse_input(grp)
            except:
                print('i: {}'.format(i))

        def cmp_mean_range(feat_vec_lst):
            if len(feat_vec_lst) == 0:
                return (0, 1)
            else:
                total_vec = np.concatenate(feat_vec_lst)
                return (np.mean(total_vec, axis=0),
                        np.max(total_vec, axis=0) + np.finfo(np.float32).eps)

        mean_range_dict = {operator: cmp_mean_range(feat_vec_col[operator]) \
                           for operator in all_dicts}
        return mean_range_dict

    def get_all_plans(self, fname):
        """
            Parse from RDBLab.data file

            Args:
            - fname: the name of data file to be parsed

            Returns:
            - jss: a sanitized list of dictionary, one per query, parsed from RDBLab.the input data file
        """
        pattern_num = re.compile(r'\d+.?\d*')
        jss = []

        with open(fname, "r") as f:
            lines = [line for line in f.readlines()]
            lineid = 0
            while lineid < len(lines):
                if ' CST [' not in lines[lineid]:
                    lineid += 1
                    continue

                while lineid < len(lines) and ' CST [' in lines[lineid]:
                    if 'duration' in lines[lineid]:
                        
                        duration = pattern_num.findall(lines[lineid])[-1]
                        
                    else:
                        lineid += 1
                        continue
                    plan_strs = []
                    lineid += 1
                    while lineid < len(lines) and ' CST [' not in lines[lineid]:
                        plan_strs.append(lines[lineid])
                        lineid += 1
                    if plan_strs != []:
                        plan_obj = json.loads(s=''.join(plan_strs))
                        plan_obj['Plan']['Planning Time'] = float(duration) - plan_obj['Plan']['Actual Total Time']
                        jss.append(plan_obj['Plan'])
        return jss

    def grouping(self, data):
        """
            Groups the queries by their query plan structure

            Args:
            - data: a list of dictionaries, each being a query from RDBLab.the dataset

            Returns:
            - enum    : a list of same length as data, containing the group indexes for each query in data
            - counter : number of distinct groups/templates
        """

        def hash(plan_dict):
            res = plan_dict['Node Type']
            if 'Plans' in plan_dict:
                for chld in plan_dict['Plans']:
                    res += hash(chld)
            return res

        counter = 0
        string_hash = []
        enum = []
        for plan_dict in data:
            string = hash(plan_dict)
            try:
                idx = string_hash.index(string)
                enum.append(idx)
            except:
                idx = counter
                counter += 1
                enum.append(idx)
                string_hash.append(string)
        # print(f"{counter} distinct templates identified")
        # print(f"Operators: {string_hash}")
        assert (counter > 0)
        return enum, counter

    def get_input(self, data):  # Helper for sample_data
        """
            Vectorize the input of a list of queries that have the same plan structure (of the same template/group)

            Args:
            - data: a list of plan_dict, each plan_dict correspond to a query plan in the dataset;
                    requires that all plan_dicts is of the same query template/group

            Returns:
            - new_samp_dict: a dictionary, where each level has the following attribute:
                -- node_type     : name of the operator
                -- subbatch_size : number of queries in data
                -- feat_vec      : a numpy array of shape (batch_size x feat_dim) that's
                                   the vectorized inputs for all queries in data
                -- children_plan : list of dictionaries with each being an output of
                                   a recursive call to get_input on a child of current node
                -- total_time    : a vector of prediction target for each query in data
                -- is_subplan    : if the queries are subplans
        """
        new_samp_dict = {}
        new_samp_dict["node_type"] = data[0]["Node Type"]
        new_samp_dict["subbatch_size"] = len(data)

        feat_vec = np.array(
            [np.hstack(
                (self.input_func[jss["Node Type"]](jss), jss['inner_vector']))
                for jss in data])

        # normalize feat_vec
        feat_vec = (feat_vec -
                    self.mean_range_dict[new_samp_dict["node_type"]][0]) \
                   / self.mean_range_dict[new_samp_dict["node_type"]][1]

        total_time = [jss['Actual Total Time'] for jss in data]
        total_cost = [jss['Total Cost'] for jss in data]

        if 'Planning Time' in data[0]:
            plan_time = [jss['Planning Time'] for jss in data]
            new_samp_dict["plan_time"] = np.array(plan_time).astype(np.float32) / self.SCALE

        child_plan_lst = []
        if 'Plans' in data[0]:
            for i in range(len(data[0]['Plans'])):
                child_plan_dict = self.get_input([jss['Plans'][i] for jss in data])
                child_plan_lst.append(child_plan_dict)

        new_samp_dict["feat_vec"] = np.array(feat_vec).astype(np.float32)
        new_samp_dict["children_plan"] = child_plan_lst
        new_samp_dict["total_time"] = np.array(total_time).astype(np.float32) / self.SCALE
        new_samp_dict["total_cost"] = np.array(total_cost).astype(np.float32) / self.SCALE

        if 'Subplan Name' in data[0]:
            new_samp_dict['is_subplan'] = True
        else:
            new_samp_dict['is_subplan'] = False
        return new_samp_dict

    ###############################################################################
    #       Sampling subbatch data from RDBLab.the dataset; total size is batch_size     #
    ###############################################################################
    def sample_data(self, batch_size):
        """
            Randomly sample a batch of data points from RDBLab.the train dataset

            Returns:
            - parsed_input: a list of dictionaries with inputs vectorized by get_input,
                            each dictionary contains all samples in the batch that comes from RDBLab.this group
        """
        # dataset: all queries used in training
        if batch_size == 0:
            batch_size = int(self.datasize*0.4)

        samp = np.random.choice(np.arange(self.datasize), batch_size, replace=False)

        samp_group = [[[] for j in range(self.num_grps[i])]
                      for i in range(self.num_q)]
        for idx in samp:
            grp_idx = self.grp_idxes[idx]
            samp_group[idx // self.num_sample_per_q][grp_idx].append(self.dataset[idx])

        parsed_input = []
        for i, temp in enumerate(samp_group):
            for grp in temp:
                if len(grp) != 0:
                    parsed_input.append(self.get_input(grp))

        return parsed_input

    def add_knobs(self, data, knobs):
        data['knob_values'] = knobs

        if 'Plans' in data:
            for sub_tree in data['Plans']:
                self.add_knobs(sub_tree, knobs)
