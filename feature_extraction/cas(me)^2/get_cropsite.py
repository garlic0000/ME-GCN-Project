import yaml
import csv
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Tuple

# 裁剪和表情帧文件夹命令 和 规范命名的转换
ch_file_name_dict = {"disgust1": "0101", "disgust2": "0102", "anger1": "0401", "anger2": "0402",
                     "happy1": "0502", "happy2": "0503", "happy3": "0505", "happy4": "0507",
                     "happy5": "0508"}
# facebox_csv_root_path = "D:/PycharmProjects/ME-GCN-Project/feature_extraction/cas(me)^2/faceboxcsv"
facebox_csv_root_path = "/kaggle/working/data/casme_2/faceboxcsv"
os.makedirs(facebox_csv_root_path, exist_ok=True)
def record_csv(csv_path, rows):
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w') as f:
        csv_w = csv.writer(f)
        csv_w.writerows(rows)


def get_site(opt) -> None:
    """
    处理裁剪图像并将人脸框信息保存到 CSV 文件。

    参数:
        opt (Dict[str, str]): 包含裁剪图像和选定图像路径的字典。
    """
    casme_2_cropped = opt["casme_2_cropped"]
    casme_2_selectpic = opt["casme_2_selectpic"]

    for sub_item in Path(casme_2_cropped).iterdir():
        if not sub_item.is_dir():
            continue

        for type_item in sub_item.iterdir():
            if not type_item.is_dir():
                continue

            v_name = f"casme_0{sub_item.name[1:]}_{ch_file_name_dict.get(type_item.name.split('_')[0])}"
            new_dir_path = os.path.join(facebox_csv_root_path, sub_item.name, v_name)

            os.makedirs(new_dir_path, exist_ok=True)  # 更简洁的目录创建方式

            facebox_csv_path = os.path.join(new_dir_path, f"{type_item.name}.csv")
            img_path_list = glob.glob(os.path.join(str(type_item), "*.jpg"))
            print(img_path_list)
            print("dgsdfgeger")
            if img_path_list:

                facebox_list = []
                img_path_list.sort()

                for img_path in img_path_list:
                    file_name = os.path.basename(img_path)
                    original_image_path = os.path.join(casme_2_selectpic, sub_item.name, type_item.name,
                                                       f"{file_name}.jpg")

                    if os.path.exists(original_image_path):
                        facebox = get_original_site_from_cropped(img_path, original_image_path)
                        if facebox is not None:
                            facebox_list.append(facebox)

                record_csv(facebox_csv_path, facebox_list)


def example():
    """
    将裁剪的图片映射到原图中
    例子
    """
    # 加载裁剪图像和原图
    original_image = cv2.imread('/kaggle/input/casme2/selectedpic/selectedpic/s15/anger1_1/img557.jpg', 0)
    cropped_image = cv2.imread('/kaggle/input/casme2/cropped/cropped/15/anger1_1/img557.jpg', 0)

    # 初始化 ORB 特征检测器
    orb = cv2.ORB_create()

    # 检测并计算特征点
    keypoints1, descriptors1 = orb.detectAndCompute(cropped_image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(original_image, None)

    # 使用 BFMatcher 进行特征点匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 绘制匹配结果
    matched_image = cv2.drawMatches(cropped_image, keypoints1, original_image, keypoints2, matches[:10], None, flags=2)

    # 获取匹配的特征点坐标
    if len(matches) > 0:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 估算变换矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 裁剪图像的四个角点
        h, w = cropped_image.shape
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype='float32').reshape(-1, 1, 2)

        # 使用变换矩阵将角点映射回原图
        transformed_corners = cv2.perspectiveTransform(corners, M)

        # 在原图上绘制裁剪图像的边缘
        original_image_with_edges = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)  # 转换为彩色图像以绘制彩色边缘
        cv2.polylines(original_image_with_edges, [np.int32(transformed_corners)], isClosed=True, color=(0, 255, 0),
                      thickness=2)

        # 显示带边缘的原图
        plt.imshow(original_image_with_edges)
        plt.title("Original Image with Cropped Image Edges")
        plt.axis('off')
        plt.show()

        # 显示匹配结果
        plt.imshow(matched_image)
        plt.title("Feature Matches")
        plt.axis('off')
        plt.show()

        # 显示变换后的点
        point_in_cropped_image = np.array([[50, 60]], dtype='float32').reshape(-1, 1, 2)
        point_in_original_image = cv2.perspectiveTransform(point_in_cropped_image, M)

        print(f"Point in original image: {point_in_original_image}")

        # 计算裁剪区域的最小外接矩形
        x, y, w, h = cv2.boundingRect(transformed_corners)

        # 在原图中裁剪出人脸区域
        face_cropped_image = original_image[y:y + h, x:x + w]

        # 显示裁剪的人脸区域
        plt.imshow(face_cropped_image, cmap='gray')
        plt.title("Cropped Face from Original Image")
        plt.axis('off')
        plt.show()





def get_original_site_from_cropped(cropped_image_path: str, original_image_path: str) -> Optional[
    Tuple[int, int, int, int]]:
    """
    将已裁剪的图片映射到原图上，获取裁剪的位置。

    参数:
        cropped_image_path (str): 裁剪图像的文件路径。
        original_image_path (str): 原图的文件路径。

    返回:
        Optional[Tuple[int, int, int, int]]: 返回裁剪区域的左、上、右、下坐标，如果未找到则返回 None。
    """
    # 加载裁剪图像和原图
    original_image = cv2.imread(original_image_path, 0)
    cropped_image = cv2.imread(cropped_image_path, 0)

    if original_image is None or cropped_image is None:
        print("Error loading images.")
        return None

    # 初始化 ORB 特征检测器
    orb = cv2.ORB_create()

    # 检测并计算特征点
    keypoints1, descriptors1 = orb.detectAndCompute(cropped_image, None)
    keypoints2, descriptors2 = orb.detectAndCompute(original_image, None)

    # 使用 BFMatcher 进行特征点匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 获取匹配的特征点坐标
    if len(matches) > 0:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # 估算变换矩阵
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # 裁剪图像的四个角点
        h, w = cropped_image.shape
        corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype='float32').reshape(-1, 1, 2)

        # 使用变换矩阵将角点映射回原图
        transformed_corners = cv2.perspectiveTransform(corners, M)

        # 计算裁剪区域的最小外接矩形
        left, top, width, height = cv2.boundingRect(transformed_corners)
        right = left + width
        bottom = top + height
        return left, top, right, bottom

    return None


if __name__ == "__main__":
    with open("/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    get_site(opt)
