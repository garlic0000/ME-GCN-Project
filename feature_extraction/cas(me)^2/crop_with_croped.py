import os
import glob
from pathlib import Path
import cv2
from tqdm import tqdm
import yaml
import shutil
import pandas as pd
from tools import FaceDetector

"""
按照数据集中已经裁剪好的图片的尺寸进行裁剪

需要将已裁剪好的图片映射回去
将该尺寸用于每个图片 也不一定 
将需要进行裁剪的尺寸记录在文件中
"""


def get_img_count(root_path):
    count = 0
    for sub_item in Path(root_path).iterdir():
        if not sub_item.is_dir():
            continue
        for type_item in sub_item.iterdir():
            if not type_item.is_dir():
                continue
            # # 计算目录下所有 .jpg 文件的数量
            count += len(glob.glob(os.path.join(str(type_item), "*.jpg")))
    return count


def csv_to_tuple(csv_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_path, header=None)

    # 将 DataFrame 的第一行转换为一维列表
    data_as_list = df.iloc[0].tolist()

    return tuple(data_as_list)


def crop(opt):
    try:
        simpled_root_path = opt["simpled_root_path"]
        cropped_root_path = opt["cropped_root_path"]
        dataset = opt["dataset"]
        facebox_csv_root_path = "/kaggle/working/data/casme_2/faceboxcsv"
    except KeyError:
        print(f"Dataset {dataset} does not need to be cropped")
        print("terminate")
        exit(1)

    sum_count = get_img_count(simpled_root_path)
    print("img count = ", sum_count)

    if not os.path.exists(simpled_root_path):
        print(f"path {simpled_root_path} is not exist")
        exit(1)

    if not os.path.exists(cropped_root_path):
        os.makedirs(cropped_root_path)

    # 裁剪模型
    face_det_model_path = opt.get("face_det_model_path")
    face_detector = FaceDetector(face_det_model_path)

    with tqdm(total=sum_count) as tq:
        for sub_item in Path(simpled_root_path).iterdir():
            if not sub_item.is_dir():
                continue
            for type_item in sub_item.iterdir():
                if not type_item.is_dir():
                    continue
                # 在这里修改
                # s15 15_0101
                # casme_015,casme_015_0401
                # subject video_name
                # 将type_item改为别的
                # s15 casme_015
                # /kaggle/input/casme2/rawpic/rawpic/s15/15_0101disgustingteeth

                s_name = "casme_0{}".format(sub_item.name[1:])
                v_name = "casme_0{}".format(type_item.name[0:7])
                new_dir_path = os.path.join(
                    cropped_root_path, s_name, v_name)
                print(new_dir_path)
                # new_dir_path = os.path.join(
                #     cropped_root_path, sub_item.name, type_item.name)
                if not os.path.exists(new_dir_path):
                    os.makedirs(new_dir_path)
                facebox_average_path = os.path.join(facebox_csv_root_path, sub_item.name,
                                                    v_name, "facebox_average.csv")
                if not os.path.exists(facebox_average_path):
                    print(f"{facebox_average_path}不存在")
                    continue
                # 获取目录下所有 .jpg 文件的路径，并将它们存储在一个列表中
                img_path_list = glob.glob(
                    os.path.join(str(type_item), "*.jpg"))
                if len(img_path_list) > 0:
                    img_path_list.sort()
                    face_left, face_top, face_right, face_bottom = \
                        csv_to_tuple(facebox_average_path)

                    for index, img_path in enumerate(img_path_list):
                        img = cv2.imread(img_path)
                        img = img[face_top:face_bottom + 1, face_left:face_right + 1, :]
                        # # 用于调错
                        # # 检测裁剪后的图片是否能检测到人脸
                        # check_crop(img, img_path)
                        # 不写 只测试
                        cv2.imwrite(os.path.join(
                            new_dir_path,
                            f"img_{str(index + 1).zfill(5)}.jpg"), img)
                        tq.update()


if __name__ == "__main__":
    import os

    # os.environ['CUDA_VISIBLE_DEVICES']    = '3, 4'
    # 只有0可以用
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    with open("/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    crop(opt)
