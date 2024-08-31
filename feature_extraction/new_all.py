import yaml

from new_sampling import sampling
from new_crop import crop
from new_record_face_and_landmark import record_face_and_landmarks
from new_optflow import optflow
from new_feature import feature
from new_feature_segment import segment_for_train, segment_for_test
from apex_sample import apex_sampling
if __name__ == "__main__":
    with open("/kaggle/working/AUW-GCN-test/feature_extraction/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]

    import os
    # os.environ['CUDA_VISIBLE_DEVICES']    = '3, 4'
    # 只有0可以用
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # 对cas(me)^2而言 不需要进行sampling
    # print("================ sampling ================")
    # # 顶点 采样？
    # apex_sampling(opt)
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
