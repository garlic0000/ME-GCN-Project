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


