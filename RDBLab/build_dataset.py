import pickle

import RDBLab.config
from RDBLab.dataset.oltp_dataset.oltp_utils_origin import OLTPDataSet as OLTPDataSet_origin,tpcc_dim as origin_tpcc_dim
from RDBLab.dataset.oltp_dataset.oltp_utils_knob import OLTPDataSet as OLTPDataSet_knob,tpcc_dim as knob_tpcc_dim
from RDBLab.dataset.oltp_dataset.oltp_utils_serialize import OLTPDataset as OLTPDataSet_serialize
from RDBLab.dataset.oltp_dataset.oltp_utils_serialize_knob import OLTPDataset as OLTPDataSet_knob_serialize

from RDBLab.dataset.postgres_tpch_dataset.tpch_utils_origin import tpch_dim_dict as origin_tpch_dim, \
    PSQLTPCHDataSet as PSQLTPCHDataSet_origin
from RDBLab.dataset.postgres_tpch_dataset.tpch_utils_knob import tpch_dim_dict as knob_tpch_dim, \
    PSQLTPCHDataSet as PSQLTPCHDataSet_knob
from RDBLab.dataset.postgres_tpch_dataset.tpch_utils_pro import tpch_dim_dict as pro_tpch_dim, \
    PSQLTPCHDataSet as PSQLTPCHDataSet_pro
from RDBLab.dataset.postgres_tpch_dataset.tpch_utils_knob_config import tpch_dim_dict as config_tpch_dim, \
    PSQLTPCHDataSet as PSQLTPCHDataSet_config
from RDBLab.dataset.postgres_tpch_dataset.tpch_utils_serialize import \
    PSQLTPCHDataSet as PSQLTPCHDataSet_serialize
from RDBLab.dataset.postgres_tpch_dataset.tpch_utils_serialize_knob import \
    PSQLTPCHDataSet as PSQLTPCHDataSet_knob_serialize

from RDBLab.dataset.sysbench_dataset.sysbench_utils_origin import SysbenchDataset as SysbenchDataset_origin,sysbench_dim_dict as origin_sysbench_dim
from RDBLab.dataset.sysbench_dataset.sysbench_utils_knob import SysbenchDataset as SysbenchDataset_knob,sysbench_dim as knob_sysbench_dim
from RDBLab.dataset.sysbench_dataset.sysbench_utils_serialize import SysbenchDataset as SysbenchDataset_serialize
from RDBLab.dataset.sysbench_dataset.sysbench_utils_serialize_knob import SysbenchDataset as SysbenchDataset_knob_serialize

from RDBLab.dataset.job_dataset.job_utils_origin import jobDataset as jobDataset_origin,job_dim as origin_job_dim
from RDBLab.dataset.job_dataset.job_utils_knob import jobDataset as jobDataset_knob,job_dim as knob_job_dim
from RDBLab.dataset.job_dataset.job_utils_serialize import jobDataset as jobDataset_serialize
from RDBLab.dataset.job_dataset.job_utils_serialize_knob import jobDataset as jobDataset_knob_serialize

from RDBLab.utils.util import Utils

PSQLTPCH={
    "dim_dict":{
        "origin_model": origin_tpch_dim,
        "knob_model": knob_tpch_dim,
        "pro_model": pro_tpch_dim,
        "config_model": config_tpch_dim,
    },
    "dataset":{
        "origin_model": PSQLTPCHDataSet_origin,
        "knob_model": PSQLTPCHDataSet_knob,
        "pro_model": PSQLTPCHDataSet_pro,
        "config_model": PSQLTPCHDataSet_config,
        "serialize": PSQLTPCHDataSet_serialize,
        "serialize_knob": PSQLTPCHDataSet_knob_serialize,
    }
}
PSQLSysbench={
    "dim_dict":{
        "origin_model": origin_sysbench_dim,
        "knob_model": knob_sysbench_dim,
    },
    "dataset":{
        "origin_model": SysbenchDataset_origin,
        "knob_model": SysbenchDataset_knob,
        "serialize": SysbenchDataset_serialize,
        "serialize_knob": SysbenchDataset_knob_serialize,
    }
}
PSQLTPCC={
    "dim_dict":{
        "origin_model": origin_tpcc_dim,
        "knob_model": knob_tpcc_dim,
    },
    "dataset":{
        "origin_model": OLTPDataSet_origin,
        "knob_model": OLTPDataSet_knob,
        "serialize": OLTPDataSet_serialize,
        "serialize_knob": OLTPDataSet_knob_serialize,
    }
}
PSQLJOB={
    "dim_dict":{
        "origin_model": origin_job_dim,
        "knob_model": knob_job_dim,
    },
    "dataset":{
        "origin_model": jobDataset_origin,
        "knob_model": jobDataset_knob,
        "serialize": jobDataset_serialize,
        "serialize_knob": jobDataset_knob_serialize,
    }
}

DATASET_TYPE = {
    "PSQLTPCH": PSQLTPCH,
    "PSQLTPCC": PSQLTPCC,
    "PSQLSysbench": PSQLSysbench,
    "PSQLJOB": PSQLJOB
}

def build_ds(opt, mid_data_dir):

    Utils.path_build(opt.mid_data_dir)
    Utils.path_build(opt.data_structure)

    if "MSCN" in opt.save_dir:
        if "origin_model" in mid_data_dir:
            dataset = DATASET_TYPE[opt.dataset]['dataset']['serialize'](opt)
            with open(opt.mid_data_dir + '/serialize_dim_dict.pickle', 'rb') as f:
                dim_dict = pickle.load(f)
        if "knob_model" in mid_data_dir:
            dataset = DATASET_TYPE[opt.dataset]['dataset']['serialize_knob'](opt)
            with open(opt.mid_data_dir + '/serialize_knob_dim_dict.pickle', 'rb') as f:
                dim_dict = pickle.load(f)
        return dataset, dim_dict

    if "origin_model" in mid_data_dir:
        dataset = DATASET_TYPE[opt.dataset]['dataset']['origin_model'](opt)
        dim_dict = DATASET_TYPE[opt.dataset]['dim_dict']['origin_model']

    elif "knob_model" in mid_data_dir:
        dataset = DATASET_TYPE[opt.dataset]['dataset']['knob_model'](opt)
        dim_dict = DATASET_TYPE[opt.dataset]['dim_dict']['knob_model'](RDBLab.config.cost_factor_dict)

    elif "pro_model" in mid_data_dir:
        dataset = DATASET_TYPE[opt.dataset]['dataset']['pro_model'](opt)
        dim_dict = DATASET_TYPE[opt.dataset]['dim_dict']['pro_model'](RDBLab.config.cost_factor_dict)

    elif "config_model" in mid_data_dir:
        dataset = DATASET_TYPE[opt.dataset]['dataset']['config_model'](opt)
        dim_dict = DATASET_TYPE[opt.dataset]['dim_dict']['config_model'](RDBLab.config.cost_factor_dict)

    elif mid_data_dir == "mobility_model":
        dataset = DATASET_TYPE[opt.dataset]['dataset']['mobility_model'](opt)
        dim_dict = DATASET_TYPE[opt.dataset]['dim_dict']['mobility_model'](RDBLab.config.cost_factor_dict)

    return dataset, dim_dict
