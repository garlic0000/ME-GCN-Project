import os
import glob
import shutil
from pathlib import Path

import yaml
import numpy as np
import pandas as pd


def changeFilesWithCSV(opt):
    try:
        simpled_root_path = opt["simpled_root_path"]
        dataset = opt["dataset"]
    except KeyError:
        print(f"Dataset {dataset} does not need to be cropped")
        print("terminate")
        exit(1)
    # s_name_dict = {"s15": "casme_015", "s16": "casme_016", "s19": "casme_019", "s20": "casme_020",
    #                "s21": "casme_021", "s22": "casme_022", "s23": "casme_023", "s24": "casme_024",
    #                "s25": "casme_025", "s26": "casme_026", "s27": "casme_027", "s29": "casme_029",
    #                "s30": "casme_030", "s31": "casme_031", "s32": "casme_032", "s33": "casme_033",
    #                "s34": "casme_034", "s35": "casme_035", "s36": "casme_036", "s37": "casme_037",
    #                "s38": "casme_038", "s39": "casme_039"}
    # v_name_dict = {"casme_015"}
    ch_file_name_dict = {"disgust1": "0101", "disgust2": "0102", "anger1": "0401", "anger2": "0402",
                         "happy1": "0502", "happy2": "0503", "happy3": "0505", "happy4": "0507", "happy5": "0508"}
    for sub_item in Path(simpled_root_path).iterdir():
        # sub_item s14
        if not sub_item.is_dir():
            continue
        # type_item anger1_1
        for type_item in sub_item.iterdir():
            if not type_item.is_dir():
                continue
            # 获取当前
            for filename in ch_file_name_dict.keys():
                # anger1 anger1_1
                if filename in type_item.name:
                    # sssss/s14/0401
                    new_dir_path = os.path.join(
                        simpled_root_path, sub_item.name, ch_file_name_dict[filename])
                    if not os.path.exists(new_dir_path):
                        os.makedirs(new_dir_path)
                    # anger1_1  0401
                    shutil.copytree(
                        str(type_item), new_dir_path, dirs_exist_ok=True)
                    # 删除 type_item 目录及其内容 递归删除
                    shutil.rmtree(type_item)


