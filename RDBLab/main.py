import os

import RDBLab.config
from RDBLab.config import *
from RDBLab.build_dataset import build_ds
from RDBLab.build_model import build_md
from RDBLab.config import set_per_qs
from RDBLab.utils.draw_picture import scatter, bar
from RDBLab.utils.opt_parser import getParser
from RDBLab.utils.util import Utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

net = ['QPPNet', 'Net_v2']

dstype_type_dict = {
    'tpch': 'PSQLTPCH',
    'tpcc': 'PSQLTPCC',
    'sysbench': 'PSQLSysbench',
    'job': 'PSQLJOB',
}

data_dir_dict = {
    'tpch': 'tpch',
    'tpcc': 'tpcc',
    'sysbench': 'sysbench',
    'job': 'job',
}

start_epoch = {
    'tpch': 0,
    'tpcc': 0,
    'sysbench': 0,
    'job': 0,
}

end_epoch = {
    'tpch': 400,
    'tpcc': 100,
    'sysbench': 100,
    'job': 800,
}

per_qs_all = [
    [5, 100, 100, 100],  # 5*22*20=2200,100*20=2000,100*20=2000
    # [10, 200, 200, 200],  # 10*22*20=4400,200*20=4000,200*20=4000
    # [15, 300, 300, 300],
    # [20, 400, 400, 400],
    # [25, 500, 500, 500],
    # [30, 500, 500, 500],
]

def main():

    # level 1
    benchmark_types = [
        'tpch',
        # 'sysbench',
        # 'job',
        # 'tpcc',
    ]
    # level 2
    mid_data_dirs = [
        # "origin_model",
        "knob_model",
        # "knob_modelchange",
        # "knob_modelchange_shap",
        # "knob_modelchange_template",
        # "pro_model",
        # "knob_model_shap",
        # "knob_model_shap4net",
        # "knob_model_grad",
        # 'knob_model_R2',
        # 'knob_modeltemplate' + str(template_num * 1),
        # 'knob_modeltemplate' + str(template_num * 2),
        # 'knob_modeltemplate' + str(template_num * 3),
        # 'knob_modeltemplate' + str(template_num * 4),
        # 'mobility_model'
    ]

    # level 3
    model_types = [
        "QPPNet",
        # "RandomForest",
        # "Net_v2",
        # "MSCN",
        # "GradientBoosting",
    ]
    eval = True

    for per in per_qs_all:
        type = "-".join([str(per[0] * 22 * 20), str(per[1] * 20), str(per[2] * 20), str(per[3] * 20)])
        print(type)
        set_per_qs([p * 20 for p in per])

        for benchmark_type in benchmark_types:
            opt = []
            for model in mid_data_dirs:
                version = type + "/" + benchmark_type

                Utils.path_build("./" + version)

                for model_type in model_types:

                    new_md = True if model_type == 'QPPNet' or model_type == 'MSCN' else False

                    if 'mobility' in model:

                        mobility_model_dir = 'RDBLab/' + version + '/' + data_dir_dict[
                            benchmark_type] + "_mobility_" + model_type
                        opt = getParser(version=version,
                                        dataset=dstype_type_dict[benchmark_type],
                                        new_ds=False, new_md=False,
                                        mid_data_dir=mobility_model_dir,
                                        data_structure='RDBLab/' + version + '/data_structure',
                                        data_dir=data_dir_dict[benchmark_type],
                                        saved_model='/save_model_' + model_type,
                                        mode='train',
                                        knobs="0").parse_args()
                        dataset, dim_dict = build_ds(opt, model)
                        build_md(dataset, model_type, opt, dim_dict)

                        for knob in range(1, 20, 1):
                            # 2.1 不迁移的
                            opt = getParser(version=version,
                                            dataset=dstype_type_dict[benchmark_type],
                                            new_ds=False, new_md=False,
                                            mid_data_dir=mobility_model_dir,
                                            data_structure='RDBLab/' + version + '/data_structure',
                                            data_dir=data_dir_dict[benchmark_type],
                                            saved_model='/save_model_' + model_type,
                                            mode='mobility_eval',
                                            knobs=str(knob),
                                            change=True).parse_args()

                            dataset, dim_dict = build_ds(opt, model)
                            build_md(dataset, model_type, opt, dim_dict)

                            opt = getParser(version=version,
                                            dataset=dstype_type_dict[benchmark_type],
                                            new_ds=False, new_md=False,
                                            mid_data_dir=mobility_model_dir,
                                            data_structure='RDBLab/' + version + '/data_structure',
                                            data_dir=data_dir_dict[benchmark_type],
                                            saved_model='/save_model_' + model_type,
                                            mode='mobility_eval',
                                            knobs=str(knob)).parse_args()

                            dataset, dim_dict = build_ds(opt, model)
                            build_md(dataset, model_type, opt, dim_dict)

                            # 2.2 迁移的
                            opt = getParser(version=version,
                                            dataset=dstype_type_dict[benchmark_type],
                                            new_ds=False, new_md=False,
                                            mid_data_dir=mobility_model_dir,
                                            data_structure='RDBLab/' + version + '/data_structure',
                                            data_dir=data_dir_dict[benchmark_type],
                                            saved_model='/save_model_' + model_type,
                                            mode='part_train',
                                            knobs=str(knob)).parse_args()

                            build_md(dataset, model_type, opt, dim_dict)
                        break

                    elif 'knob_model_' in model:

                        filter_type = model.replace("knob_model_", "")

                        for i in range(2):
                            if os.path.exists("./2200-2000-2000-2000/" + benchmark_type + '/knob_model/save_model_'
                                              + model_type + "/1024/" + filter_type + "_values_array.pickle"):
                                version_sp = version
                            else:
                                version_sp = "2200-2000-2000-2000/" + benchmark_type

                            opt = getParser(version=version_sp,
                                            dataset=dstype_type_dict[benchmark_type],
                                            new_ds=False, new_md=False,
                                            mid_data_dir='RDBLab/' + version_sp + '/knob_model',
                                            data_structure='RDBLab/' + version_sp + '/data_structure',
                                            data_dir=data_dir_dict[benchmark_type],
                                            saved_model='/save_model_' + model_type,
                                            mode=filter_type + '_eval').parse_args()

                            if os.path.exists("./2200-2000-2000-2000/" + benchmark_type + '/knob_model/save_model_'
                                              + model_type + "/1024/" + filter_type + "_values_array.pickle"):
                                dataset, dim_dict = build_ds(opt, 'knob_model')
                                break
                            else:
                                dataset, dim_dict = build_ds(opt, 'knob_model')
                                build_md(dataset, model_type, opt, dim_dict)

                        opt = getParser(version=version,
                                        dataset=dstype_type_dict[benchmark_type],
                                        new_ds=False, new_md=False,
                                        mid_data_dir='RDBLab/' + version + '/' + model,
                                        data_structure='RDBLab/' + version + '/data_structure',
                                        data_dir=data_dir_dict[benchmark_type],
                                        saved_model='/save_model_' + model_type,
                                        mode='eval' if eval else 'train',
                                        start_epoch=start_epoch[benchmark_type],
                                        end_epoch=end_epoch[benchmark_type]).parse_args()

                        Utils.path_build(opt.mid_data_dir)
                        Utils.path_build(opt.data_structure)

                        build_md(dataset, model_type, opt, dim_dict)

                    else:
                        if "origin_model" in model:
                            opt = getParser(version=version,
                                            dataset=dstype_type_dict[benchmark_type],
                                            new_ds=False,new_md=False,
                                            mid_data_dir='RDBLab/' + version + '/' + model,
                                            data_structure='RDBLab/' + version + '/data_structure' + model.replace(
                                                "knob_model_template", "").replace("origin_model", ""),
                                            data_dir=data_dir_dict[benchmark_type],
                                            saved_model='/save_model_' + model_type,
                                            mode='eval' if eval else 'train',
                                            start_epoch=start_epoch[benchmark_type],
                                            end_epoch=end_epoch[benchmark_type]).parse_args()
                        elif 'change' not in model:
                            opt = getParser(version=version,
                                            dataset=dstype_type_dict[benchmark_type],
                                            new_ds=False,new_md=False,
                                            mid_data_dir='RDBLab/' + version + '/' + model,
                                            data_structure='RDBLab/' + version + '/data_structure' + model.replace(
                                                "knob_model", "").replace("origin_model", "").replace("pro_model", ""),
                                            data_dir=data_dir_dict[benchmark_type],
                                            saved_model='/save_model_' + model_type,
                                            mode='eval' if eval else 'train',
                                            start_epoch=start_epoch[benchmark_type],
                                            end_epoch=end_epoch[benchmark_type]).parse_args(args=[])
                        else:
                            temp_model = model.replace("_shap", "")
                            opt = getParser(version=version,
                                            dataset=dstype_type_dict[benchmark_type],
                                            new_ds=False, new_md=False,
                                            mid_data_dir='RDBLab/' + version + '/' + temp_model,
                                            data_structure='RDBLab/' + version + '/data_structure_' + temp_model
                                            .replace("knob_model", ""),
                                            data_dir=data_dir_dict[benchmark_type] + "_i7",
                                            saved_model='/save_model_' + model_type,
                                            mode='change_train',
                                            start_epoch=start_epoch[benchmark_type],
                                            end_epoch=end_epoch[benchmark_type]).parse_args()

                            dataset, dim_dict = build_ds(opt, model)

                            opt = getParser(version=version,
                                            dataset=dstype_type_dict[benchmark_type],
                                            new_ds=False, new_md=False,
                                            mid_data_dir='RDBLab/' + version + '/' + model,
                                            data_structure='RDBLab/' + version + '/data_structure_' + model
                                            .replace("knob_model", ""),
                                            data_dir=data_dir_dict[benchmark_type] + "_i7",
                                            saved_model='/save_model_' + model_type,
                                            mode='change_train',
                                            start_epoch=start_epoch[benchmark_type],
                                            end_epoch=end_epoch[benchmark_type]).parse_args()

                            Utils.path_build(opt.mid_data_dir)
                            build_md(dataset, model_type, opt, dim_dict)
                            scatter(opt, model_type)
                            continue

                        dataset, dim_dict = build_ds(opt, model)
                        print(dim_dict)
                        build_md(dataset, model_type, opt, dim_dict)

                    # scatter(opt, model_type)
                    # bar(opt, model_type)
