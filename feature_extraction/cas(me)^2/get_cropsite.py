import yaml
import csv
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
import shutil
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


def check():
    """
    可能是croped 中的路径不存在 但 rawpic的路径存在
    """
    if os.path.exists("/kaggle/working/data/casme_2/faceboxcsv/s15/casme_015_0508/"):
        print("/kaggle/working/data/casme_2/faceboxcsv/s15/casme_015_0508/ 存在")
    else:
        print("/kaggle/working/data/casme_2/faceboxcsv/s15/casme_015_0508/ 不存在")

    if os.path.exists("/kaggle/working/data/casme_2/faceboxcsv/s24/casme_024_0507/"):
        print("/kaggle/working/data/casme_2/faceboxcsv/s24/casme_024_0507/ 存在")
    else:
        print("/kaggle/working/data/casme_2/faceboxcsv/s24/casme_024_0507/ 不存在")
    # 原数据集的问题
    if os.path.exists("/kaggle/working/data/casme_2/faceboxcsv/s19/casme_023_0502/"):
        print("/kaggle/working/data/casme_2/faceboxcsv/s19/casme_023_0502/ 存在")
    else:
        print("/kaggle/working/data/casme_2/faceboxcsv/s19/casme_023_0502/ 不存在")
    # 正常的数据是否存在
    if os.path.exists("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0502/"):
        print("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0502/ 存在")
    else:
        print("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0502/ 不存在")

    if os.path.exists("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0503/"):
        print("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0503/ 存在")
    else:
        print("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0503/ 不存在")

    if os.path.exists("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0507/"):
        print("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0507/ 存在")
    else:
        print("/kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0507/ 不存在")


def copy_contents(src_dir, dst_dir):
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)

    # 遍历 src_dir 中的所有内容
    for item in os.listdir(src_dir):
        src_item = os.path.join(src_dir, item)
        dst_item = os.path.join(dst_dir, item)
        shutil.copy2(src_item, dst_item)

def get_patch():
    """
    有一些有表情的图片帧 但是数据集没有进行裁剪
    比如 对于015_0508 将s15目录所有的csv文件集合起来 复制到 没有的文件中
    s19 这个另外处理
    现在 目录下还没有 facebox_average.csv
    /kaggle/working/data/casme_2/faceboxcsv/s15/casme_015_0508/facebox_average.csv不存在
    /kaggle/working/data/casme_2/faceboxcsv/s24/casme_024_0507/facebox_average.csv不存在
    /kaggle/working/data/casme_2/faceboxcsv/s19/casme_023_0502/facebox_average.csv不存在
    /kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0503/facebox_average.csv不存在
    /kaggle/working/data/casme_2/faceboxcsv/s23/casme_023_0507/facebox_average.csv不存在
    """
    s15_img_path = os.path.join(facebox_csv_root_path, "s15")
    s23_img_path = os.path.join(facebox_csv_root_path, "s23")
    s24_img_path = os.path.join(facebox_csv_root_path, "s24")
    for sub_item in Path(s15_img_path).iterdir():
        if not sub_item.is_dir() or sub_item.name == "casme_015_0508":
            continue
        copy_contents(str(sub_item), os.path.join(s15_img_path, "casme_015_0508"))
    for sub_item in Path(s23_img_path).iterdir():
        if not sub_item.is_dir() or sub_item.name in ["casme_023_0503", "casme_023_0507"]:
            continue
        copy_contents(str(sub_item), os.path.join(s23_img_path, "casme_023_0503"))
        copy_contents(str(sub_item), os.path.join(s23_img_path, "casme_023_0507"))
    for sub_item in Path(s24_img_path).iterdir():
        if not sub_item.is_dir() or sub_item.name == "casme_024_0507":
            continue
        copy_contents(str(sub_item), os.path.join(s24_img_path, "casme_024_0507"))


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
            # sub_item.name 不带s
            v_name = f"casme_0{sub_item.name}_{ch_file_name_dict.get(type_item.name.split('_')[0])}"
            new_dir_path = os.path.join(facebox_csv_root_path, f"s{sub_item.name}", v_name)

            os.makedirs(new_dir_path, exist_ok=True)  # 更简洁的目录创建方式

            facebox_csv_path = os.path.join(new_dir_path, f"{type_item.name}.csv")
            # str(type_item) 是一个完整路径 type_item.name是路径最后一个文件或文件夹的名称
            img_path_list = glob.glob(os.path.join(str(type_item), "*.jpg"))
            facebox_list = []
            if img_path_list:
                img_path_list.sort()

                for img_path in img_path_list:
                    file_name = os.path.basename(img_path)
                    # selectpic 有s cropped 没有s
                    # /kaggle/input/casme2/selectedpic/selectedpic/s19/happy4_1/img2977.jpg.jpg
                    # file_name 不需要.jpg
                    original_image_path = os.path.join(casme_2_selectpic, f"s{sub_item.name}", type_item.name,
                                                       file_name)
                    if os.path.exists(original_image_path):
                        facebox = get_original_site_from_cropped(img_path, original_image_path)
                        if facebox is not None:
                            facebox_list.append(facebox)

                record_csv(facebox_csv_path, facebox_list)
            # 测试
            if len(facebox_list) == 0:
                print(v_name)
                print(facebox_list)
                print(f"{type_item.name}.csv")
    # 有一些路径 cropped不存在 但rawpic存在
    print("打补丁")
    get_patch()
    check()


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
    如果精度和鲁棒性是最关注的（例如图像匹配的准确性要求很高、复杂场景、光照变化较大），SIFT是更好的选择。
    如果速度是主要考虑的因素（例如实时应用场景），SURF可能更合适。
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

    # 使用 SIFT 特征检测器
    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        print("SIFT is not available in this version of OpenCV, falling back to ORB.")
        sift = cv2.ORB_create()

    # 检测并计算特征点和描述符
    keypoints1, descriptors1 = sift.detectAndCompute(cropped_image, None)
    keypoints2, descriptors2 = sift.detectAndCompute(original_image, None)

    # 检查是否有足够的特征点进行匹配
    if descriptors1 is None or descriptors2 is None:
        print("No descriptors found in one of the images.")
        return None
    if len(keypoints1) < 4 or len(keypoints2) < 4:
        print("Not enough keypoints found for reliable matching.")
        return None

    # 使用 FLANN 匹配器进行特征点匹配
    index_params = dict(algorithm=1, trees=5)  # 使用 KD-Tree
    search_params = dict(checks=50)  # 检查次数
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 只保留好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # Lowe's ratio test
            good_matches.append(m)

    if len(good_matches) < 4:
        print("Not enough good matches found.")
        return None

    # 获取匹配的特征点坐标
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 估算变换矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if M is None:
        print("Homography estimation failed.")
        return None

    # 裁剪图像的四个角点
    h, w = cropped_image.shape
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype='float32').reshape(-1, 1, 2)

    # 使用变换矩阵将角点映射回原图
    transformed_corners = cv2.perspectiveTransform(corners, M)

    # 计算裁剪区域的最小外接矩形
    left, top, width, height = cv2.boundingRect(transformed_corners)
    right = left + width
    bottom = top + height

    # 获取原图的尺寸
    original_h, original_w = original_image.shape

    # 检查是否超出边界，如果超出则返回 None
    if left < 0 or top < 0 or right > original_w or bottom > original_h:
        print("Transformed region exceeds original image bounds.")
        return None

    return left, top, right, bottom


if __name__ == "__main__":
    with open("/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    get_site(opt)
