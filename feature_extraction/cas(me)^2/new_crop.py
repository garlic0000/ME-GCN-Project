import os
import glob
from pathlib import Path
import cv2
from tqdm import tqdm
import yaml
import shutil

from tools import FaceDetector
"""
CROPPED_SIZE: 500
1.使用数据集自带的剪切好的图片

"""

def get_img_count(root_path, dataset):
    count = 0
    for sub_item in Path(root_path).iterdir():
        if not sub_item.is_dir():
            continue
        for type_item in sub_item.iterdir():
            if not type_item.is_dir():
                continue
            if (dataset == "samm_25"
                    and (type_item.name == "032_3"
                         or type_item.name == "032_6")):
                continue
            # # 计算目录下所有 .jpg 文件的数量
            count += len(glob.glob(os.path.join(str(type_item), "*.jpg")))
    return count

# def check_crop(img, img_path):
#     """
#     检测裁剪后的图片是否能再次检测人脸
#     用于调整裁剪尺寸用
#     """
#     face_detector = FaceDetector()
#     try:
#         face_left, face_top, face_right, face_bottom = \
#             face_detector.cal(img)
#     except Exception:
#         print("\n")
#         print("该路径的图片裁剪出错")
#         print(img_path)
#         face_detector.info(img)




def crop(opt):
    try:
        simpled_root_path = opt["simpled_root_path"]
        cropped_root_path = opt["cropped_root_path"]
        dataset = opt["dataset"]
    except KeyError:
        print(f"Dataset {dataset} does not need to be cropped")
        print("terminate")
        exit(1)

    sum_count = get_img_count(simpled_root_path, dataset)
    print("img count = ", sum_count)

    if not os.path.exists(simpled_root_path):
        print(f"path {simpled_root_path} is not exist")
        exit(1)

    if not os.path.exists(cropped_root_path):
        os.makedirs(cropped_root_path)

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
                # new_dir_path = os.path.join(
                #     cropped_root_path, sub_item.name, type_item.name)
                if not os.path.exists(new_dir_path):
                    os.makedirs(new_dir_path)
                # there will be some problem when crop face from 032_3 032_6.
                # These two directory should be copied to croped directory
                # directly.
                # 获取目录下所有 .jpg 文件的路径，并将它们存储在一个列表中
                img_path_list = glob.glob(
                    os.path.join(str(type_item), "*.jpg"))
                if len(img_path_list) > 0:
                    img_path_list.sort()
                    for index, img_path in enumerate(img_path_list):
                        img = cv2.imread(img_path)
                        # 为什么index == 0
                        # 对第一个图像进行剪切
                        # 将之后的图像进行对齐
                        if index == 0:
                            # h, w, c = img.shape
                            face_left, face_top, face_right, face_bottom = \
                                face_detector.cal(img)
                            # print("\n")
                            # # 输出视频文件夹的名称
                            # d_path = os.path.dirname(img_path)
                            # print(d_path)
                            # print(new_dir_path)
                            # 对上 下 左 右 进行填充或裁剪
                            # padding_top, padding_bottom, padding_left, padding_right = \
                            #     solve_img_size(sub_item, type_item)
                            # clip_top = face_top - padding_top
                            # clip_bottom = face_bottom + padding_bottom
                            # clip_left = face_left - padding_left
                            # clip_right = face_right + padding_right
                            # # 对s27的处理
                            # if padding_top == -1:
                            #     clip_top = 0
                            clip_left = face_left
                            clip_right = face_right
                            clip_top = face_top
                            clip_bottom = face_bottom
                            # 之后所有的图片都按照这个尺寸进行剪切
                        # 保证光流提取时 图片的尺寸一致
                        # 在进行填充时可能会超过图片尺寸
                        # 进行测试
                        # if clip_top < 0 or clip_bottom < 0 or clip_left < 0 or clip_right < 0:
                        #     print(clip_top, clip_bottom, clip_left, clip_right)
                        #     continue
                        img = img[clip_top:clip_bottom + 1,
                                  clip_left:clip_right + 1, :]
                        # # 用于调错
                        # # 检测裁剪后的图片是否能检测到人脸
                        # check_crop(img, img_path)
                        # 不写 只测试
                        cv2.imwrite(os.path.join(
                                    new_dir_path,
                                    f"img_{str(index+1).zfill(5)}.jpg"), img)
                        # 有的路径下的图片为空 所以不是一张一张进行更新
                        # 像是一个列表一个列表的更新
                        # 但是统计到11409张还不是11409
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
