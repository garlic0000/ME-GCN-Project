import os
import glob
from pathlib import Path
from tqdm import tqdm
from tools import get_micro_expression_average_len

import yaml
# denseflow --help

"""
GPU optical flow extraction.
Usage: denseflow [params] input 

	-a, --algorithm (value:tvl1)
		optical flow algorithm (nv/tvl1/farn/brox)
	-b, --bound (value:32)
		maximum of optical flow
	--cf, --classFolder
		outputDir/class/video/flow.jpg
	-f, --force
		regardless of the marked .done file
	-h, --help (value:true)
		print help message
	--if, --inputFrames
		inputs are frames
	--newHeight, --nh (value:0)
		new height
	--newShort, --ns (value:0)
		short side length
	--newWidth, --nw (value:0)
		new width
	-o, --outputDir (value:.)
		root dir of output
	-s, --step (value:0)
		right - left (0 for img, non-0 for flow)
	--saveType, --st (value:jpg)
		save format type (png/h5/jpg)
	-v, --verbose
		verbose

	input
		filename of video or folder of frames or a list.txt of those
"""

def get_dir_count(root_path):
    count = 0
    for sub_item in Path(root_path).iterdir():
        if sub_item.is_dir():
            for type_item in sub_item.iterdir():
                if type_item.is_dir():
                    if len(glob.glob(
                            os.path.join(str(type_item), "*.jpg"))) > 0:
                        count += 1
    return count

"""
光流特征是整张图片提取？
然后再剪切ROIs？

为什么不是先剪切AUs 形成ROIs之后再从中提取？

这样剪切的ROIs不好对齐吗？

"""

def optflow(opt):
    cropped_root_path = opt["cropped_root_path"]
    optflow_root_path = opt["optflow_root_path"]
    #anno_csv_path = opt["anno_csv_path"]

    if not os.path.exists(cropped_root_path):
        print(f"path {cropped_root_path} is not exist")
        exit(1)
    if not os.path.exists(optflow_root_path):
        os.makedirs(optflow_root_path)

    dir_count = get_dir_count(cropped_root_path)
    print("flow count = ", dir_count)

    opt_step = 1 # int(get_micro_expression_average_len(anno_csv_path) // 2)

    with tqdm(total=dir_count) as tq:
        for sub_item in Path(cropped_root_path).iterdir():
            if sub_item.is_dir():
                new_sub_dir_path = os.path.join(
                    optflow_root_path, sub_item.name)
                if not os.path.exists(new_sub_dir_path):
                    os.makedirs(new_sub_dir_path)
                for type_item in sub_item.iterdir():
                    if type_item.is_dir():
                        # sh: 1: denseflow: not found
                        # 需要安装desenflow
                        # 处理视频 获取光流特征
                        # 输出处理的当前路径

                        cmd = (f'denseflow "{str(type_item)}" -b=10 -a=tvl1 '
                               f'-s={opt_step} -if -o="{new_sub_dir_path}"')
                        print(os.path.join(cropped_root_path, sub_item, type_item))
                        print("\n")
                        os.system(cmd)
                        tq.update()


if __name__ == "__main__":
    with open("/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
    optflow(opt)
