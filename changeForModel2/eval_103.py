import torch
import opts
from model2.model_103 import AUwGCNWithMultiHeadGATAndTCN
from datasets import LOSO_DATASET
import os
import yaml
import multiprocessing
import warnings

from utils.eval_utils import eval_single_epoch, nms_single_epoch, calculate_epoch_metrics, \
    choose_best_epoch

if __name__ == '__main__':
    import opts

    # # /opt/conda/lib/python3.10/multiprocessing/popen_fork.py: 66: RuntimeWarning: os.fork()
    # # was called.os.fork() is incompatible
    # # with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
    # # self.pid = os.fork()
    #
    # # 使用spawn而不是fork: Python的multiprocessing库默认使用fork来创建子进程
    # # 但在多线程应用中可能会出现问题。你可以通过设置multiprocessing的启动方式为spawn来避免这种问题。
    # multiprocessing.set_start_method("spawn")
    # 使用spawn太慢了
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="os.fork() was called.")

    args = opts.parse_args()

    # load config & params.
    with open("/kaggle/working/ME-GCN-Project/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        if args.dataset is not None:
            dataset = args.dataset
        else:
            dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
        opt['dataset'] = dataset

    subject = args.subject

    # update opt. according to args.
    opt['output_dir_name'] = os.path.join(args.output, subject)  # ./debug/casme_016
    opt['model_save_root'] = os.path.join(opt['output_dir_name'], 'models')  # ./debug/casme_016/models/
    opt['subject'] = subject

    # define dataset & loader
    dataset = LOSO_DATASET(opt, 'test', subject)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt['batch_size'],
                                             shuffle=False,
                                             # num_workers=8
                                             num_workers=4,
                                             pin_memory=True,
                                             drop_last=False)

    # define and load model
    device = opt['device'] if torch.cuda.is_available() else 'cpu'
    model = AUwGCNWithMultiHeadGATAndTCN(opt)
    model = model.to(device)
    print("Starting evaluating...\n")
    print("Using GPU: {} \n".format(device))

    # evaluate each ckpt's model and generate proposals
    # after generating proposals, NMS to reduce overlapped proposals
    epoch_begin = opt['epoch_begin']
    for epoch in range(opt['epochs']):
        if epoch >= epoch_begin:
            with torch.no_grad():
                weight_file = os.path.join(
                    opt["model_save_root"],
                    "checkpoint_epoch_" + str(epoch).zfill(3) + ".pth.tar")
                # 这里为什么是cpu？
                # 如果你的训练模型是在 GPU 上训练的，但在运行时没有可用的 GPU，
                # 或者你希望确保代码能够在没有 GPU 的环境中运行，那么将模型加载到 CPU 是一种合理的选择
                # 有时将模型加载到 CPU 上可以节省 GPU 内存，尤其是在模型调试或推理时
                # weights_only=True？
                # ou are using torch.load with weights_only=False (the current default value),
                # which uses the default pickle module implicitly.
                # It is possible to construct malicious pickle data
                # which will execute arbitrary code during unpickling
                # 当前 torch.load() 函数默认会使用 weights_only=False，这意味着在加载模型时不仅会加载模型权重，
                # 还会加载其他可能嵌入的对象。然而，加载不受信任的文件时，可能存在安全风险，
                # 因为恶意代码可以通过 pickle 的反序列化机制在未受控制的情况下执行
                # 在未来版本中，weights_only 参数将默认为 True，这意味着只加载模型权重，不加载潜在的其他对象
                checkpoint = torch.load(weight_file,
                                        map_location=torch.device("cpu"), weights_only=True)
                model.load_state_dict(checkpoint['state_dict'])
                eval_single_epoch(opt, model, dataloader, epoch, device)

                nms_single_epoch(opt, epoch)

    # print("Calculate metrics of all the epochs\n")
    # calculate metrics of all the epochs
    calculate_epoch_metrics(opt)
    # print("epoch_metrics csv save in {0}/epoch_metrics.csv\n".format(opt['output_dir_name']))

    # print("Choose the best epoch according to criterion\n")
    # choose the best epoch according to criterion
    choose_best_epoch(opt, criterion='all_f1')
    # print("best_res csv save in {0}/best_res.csv\n".format(opt['output_dir_name']))

    print("Finish evaluating!\n")
