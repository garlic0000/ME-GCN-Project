import os
import glob
import csv
from pathlib import Path

import yaml
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools import LandmarkDetector, FaceDetector


def get_img_count(cropped_root_path):
    """
    获取图片数量
    """
    count = 0
    for sub_item in Path(cropped_root_path).iterdir():
        if sub_item.is_dir():
            for type_item in sub_item.iterdir():
                if type_item.is_dir():
                    # # 计算目录下所有 .jpg 文件的数量
                    count += len(
                        glob.glob(os.path.join(str(type_item), "*.jpg")))
    return count


def record_csv(csv_path, rows):
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w') as f:
        csv_w = csv.writer(f)
        csv_w.writerows(rows)

# 不使用cv进行弹出对话框的形式显示
# def show_img(img, face, landmarks):
#     for i in range(len(landmarks) // 2):
#         cv2.circle(img, (landmarks[i], landmarks[i + 68]), 1, (0, 0, 255), 4)
#     cv2.rectangle(img, (face[0], face[1]), (face[2], face[3]), (0, 0, 255), 1)
#     cv2.imshow("landmark", img)
#     cv2.waitKey(0)
def show_img(img, face, landmarks):
    # 将BGR格式转换为RGB格式，cv2默认是BGR，而plt需要RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 绘制 landmarks
    for i in range(len(landmarks) // 2):
        plt.plot(landmarks[i], landmarks[i + 68], 'ro', markersize=2)

    # 绘制人脸框
    plt.gca().add_patch(plt.Rectangle(
        (face[0], face[1]), face[2] - face[0], face[3] - face[1],
        edgecolor='red', facecolor='none', linewidth=1))

    # 显示图像
    plt.imshow(img_rgb)
    plt.title('Landmarks and Face')
    plt.axis('off')  # 不显示坐标轴
    plt.show()


def record_face_and_landmarks(opt):
    cropped_root_path = opt["cropped_root_path"]

    if not os.path.exists(cropped_root_path):
        print(f"path {cropped_root_path} is not exist")
        exit(1)

    sum_count = get_img_count(cropped_root_path)
    print("img count = ", sum_count)
    # face_det_model_path = "/kaggle/input/checkpoint/pytorch/default/1/retinaface_Resnet50_Final.pth"
    # 加载模型
    face_detector = FaceDetector()
    landmark_model_path = '/kaggle/input/dlib/pytorch/default/1/shape_predictor_68_face_landmarks.dat'
    landmark_detector = LandmarkDetector(landmark_model_path)

    with tqdm(total=sum_count) as tq:
        for sub_item in Path(cropped_root_path).iterdir():
            if not sub_item.is_dir():
                continue
            for type_item in sub_item.iterdir():
                if not type_item.is_dir():
                    continue
                img_path_list = glob.glob(
                    os.path.join(str(type_item), "*.jpg"))
                if len(img_path_list) > 0:
                    img_path_list.sort()
                    rows_face = []
                    rows_landmark = []
                    csv_face_path = os.path.join(str(type_item), "face.csv")
                    csv_landmark_path = os.path.join(
                        str(type_item), "landmarks.csv")
                    for index, img_path in enumerate(img_path_list):
                        img = cv2.imread(img_path)
                        try:
                            # 对已经进行人脸裁剪的图像进行检测
                            left, top, right, bottom = face_detector.cal(img)
                            x_list, y_list = landmark_detector.cal(img)
                            # 测试用
                            if index == 0:
                                print("\n")
                                dir_path = os.path.dirname(img_path)
                                print(dir_path)
                        except Exception:
                            # subject: s35, em_type: {type_item.name}, index: {index}
                            print("\n")
                            print("该路径的图片裁剪和关键点检测出错")
                            print(img_path)
                            face_detector.info(img)
                            landmark_detector.info(img)
                            break
                        # print(f"face_width: {right-left+1}")
                        # print(f"face_height: {bottom-top+1}")
                        # print(f"image_width: {img.shape[1]}")
                        # print(f"image_width: {img.shape[0]}")
                        # show_img(img, (left, top, right, bottom),
                        #          x_list + y_list)

                        rows_face.append((left, top, right, bottom))
                        rows_landmark.append(x_list + y_list)
                        # 这里是一张一张的更新
                        tq.update()
                    if len(rows_face) == len(img_path_list):
                        # 所有的脸都检测成功才能记载，而且只记载一次
                        record_csv(csv_face_path, rows_face)
                        record_csv(csv_landmark_path, rows_landmark)


if __name__ == "__main__":
    with open("/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    record_face_and_landmarks(opt)
