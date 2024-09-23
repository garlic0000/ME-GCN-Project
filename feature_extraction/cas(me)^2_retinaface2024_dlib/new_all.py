import yaml
from changeFilesWithAnnoCSV import changeFilesWithCSV
from new_sampling import sampling
from new_crop import crop
from new_record_face_and_landmark import record_face_and_landmarks
from new_optflow import optflow
from new_feature import feature
from new_feature_segment import segment_for_train, segment_for_test
from apex_sample import apex_sampling
if __name__ == "__main__":
    with open("/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]

    import os
    # os.environ['CUDA_VISIBLE_DEVICES']    = '3, 4'
    # 只有0可以用
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("================ sampling ================")
    """
    对数据集进行采样
    从已经被拆解为图片帧中的视频中选取关键帧(表情帧)
    所以原始数据是图片
    由于数据集CAS(ME)^2中存在已剪切好的标注的关键帧
    可直接进行面部关键点处理
    但是  使用数据库裁剪的检测不出人脸
    """
    # apex_sampling(opt)
    # print("处理文件夹名称")
    # changeFilesWithCSV(opt)
    print("================ crop ================")
    crop(opt)
    print("================ record ================")
    record_face_and_landmarks(opt)
    print("================ optical flow ================")
    optflow(opt)
    print("================ feature ================")
    feature(opt)
    print("================ feature segment ================")
    segment_for_train(opt)
    segment_for_test(opt)
