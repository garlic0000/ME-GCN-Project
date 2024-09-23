import os
import shutil
import glob
from pathlib import Path
from tqdm import tqdm

import yaml
import pandas as pd


def get_img_sampling_count(root_path, sampling_ratio):
    sum_count = 0
    for sub_item in Path(root_path).iterdir():
        if sub_item.is_dir():
            for type_item in sub_item.iterdir():
                if type_item.is_dir():
                    sum_count += 3
    return sum_count


def apex_sampling(opt):
    try:
        # 原始数据路径 被分解为图片帧的视频
        # 原始数据路径是 图片
        original_root_path = opt["original_root_path"]
        # 保存采样的路径
        simpled_root_path = opt["simpled_root_path"]
        # 读取注释文件
        # original_anno_csv_path = opt["original_anno_csv_path"]
        anno_csv_path = opt["anno_csv_path"]
        SAMPLING_RATIO = opt["SAMPLING_RATIO"]
    except KeyError:
        # print(f"Dataset {dataset} does not need to be sampled")
        print("Dataset {} does not need to be sampled".format(opt["dataset"]))
        print("terminate")
        # exit(1)
        return

    if not os.path.exists(original_root_path):
        print(f"path {original_root_path} is not exist")
        # exit(1)
        return

    sum_count = get_img_sampling_count(original_root_path, SAMPLING_RATIO)
    print("img count = ", sum_count)

    if not os.path.exists(simpled_root_path):
        os.makedirs(simpled_root_path)
    # 对于cas(me)^2而言
    # raise KeyError(key) from err
    # KeyError: 'Filename'
    df = pd.read_csv(anno_csv_path)
    print("打印列名")
    print(df.columns)
    # sampling img
    with tqdm(total=sum_count) as tq:
        for sub_item in Path(original_root_path).iterdir():
            if not sub_item.is_dir():
                continue
            for type_item in sub_item.iterdir():
                if not sub_item.is_dir():
                    continue

                # 创建新的目录
                new_dir_path = os.path.join(simpled_root_path, sub_item.name,
                                            type_item.name)
                if not os.path.exists(new_dir_path):
                    os.makedirs(new_dir_path)
               
                img_path_list = glob.glob(
                    os.path.join(str(type_item), "*.jpg"))
                img_path_list.sort()

                label = df[(df['Filename']==type_item.name) & (df['Subject']==int(sub_item.name[3:5]))]
                apex = int(label['ApexFrame'])
                on = int(label['OnsetFrame'])
                off = int(label['OffsetFrame'])
                #import pdb;pdb.set_trace()
                if apex-on > off-apex:
                    sampling_index_list = [0, int((apex-on)/2), apex-on, off-on]
                else:
                    sampling_index_list = [0, apex-on, int((off-apex)/2), off-on]
                
                for i, sampling_index in enumerate(sampling_index_list):
                    shutil.copy(
                        img_path_list[sampling_index],
                        os.path.join(
                            new_dir_path, "img_" + str(i).zfill(5) + ".jpg"))
                    tq.update()

    # sampling annotaion file
    # df = pd.read_csv(original_anno_csv_path)
    # df["start_frame"] = df["start_frame"] // SAMPLING_RATIO
    # df["apex_frame"] = df["apex_frame"] // SAMPLING_RATIO
    # df["end_frame"] = df["end_frame"] // SAMPLING_RATIO
    # df.to_csv(anno_csv_path, index=False)


if __name__ == "__main__":

    with open("/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    apex_sampling(opt)
