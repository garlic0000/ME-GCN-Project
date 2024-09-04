import torch
import torch.nn as nn
import os
import numpy as np
from datasets import LOSO_DATASET
from model import AUwGCN
from torch.utils.tensorboard import SummaryWriter
from utils.train_utils import configure_optimizers
from utils.loss_func import _probability_loss, MultiCEFocalLoss_New
from functools import partial
import argparse
import yaml
# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  # fix random seed for CPU
    if torch.cuda.is_available():  # fix random seed for GPU
        torch.cuda.manual_seed(seed)  # set for current GPU
        torch.cuda.manual_seed_all(seed)  # set for all GPUs
    np.random.seed(seed)  # fix random seed for random number generation
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # Set True when GPU available
    torch.backends.cudnn.deterministic = True  # fix architecture

# for reproduction, same as orig. paper setting
same_seeds(1)

loss_list = []
# keep track of statistics
class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.correct = 0
        self.total = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    def avg(self):
        return self.sum / self.count

    def update_accuracy(self, preds, labels):
        _, predicted = torch.max(preds, 1)
        self.correct += (predicted == labels).sum().item()
        self.total += labels.size(0)

    def accuracy(self):
        return self.correct / self.total





def train(opt, data_loader, model, optimizer, epoch, device, writer):
    model.train()

    # 损失率和准确率积累器
    loss_am = AverageMeter()
    bi_apex_acc_am = AverageMeter()
    bi_action_acc_am = AverageMeter()
    micro_se_acc_am = AverageMeter()
    macro_se_acc_am = AverageMeter()

    for batch_idx, (feature, micro_apex_score, macro_apex_score,
                    micro_action_score, macro_action_score,
                    micro_start_end_label, macro_start_end_label) in enumerate(data_loader):

        # forward pass
        feature = feature.to(device)
        micro_apex_score = micro_apex_score.to(device)
        macro_apex_score = macro_apex_score.to(device)
        micro_action_score = micro_action_score.to(device)
        macro_action_score = macro_action_score.to(device)
        micro_start_end_label = micro_start_end_label.to(device)
        macro_start_end_label = macro_start_end_label.to(device)

        output_probability = model(feature)
        STEP = int(opt["RECEPTIVE_FILED"] // 2)
        output_probability = output_probability[:, :, STEP:-STEP]

        output_micro_apex = output_probability[:, 6, :]
        output_macro_apex = output_probability[:, 7, :]
        output_micro_action = output_probability[:, 8, :]
        output_macro_action = output_probability[:, 9, :]
        output_micro_start_end = output_probability[:, 0:0 + 3, :]
        output_macro_start_end = output_probability[:, 3:3 + 3, :]

        # 二分类损失
        loss_micro_apex = bi_loss_apex(output_micro_apex, micro_apex_score)
        loss_macro_apex = bi_loss_apex(output_macro_apex, macro_apex_score)
        loss_micro_action = bi_loss_action(output_micro_action, micro_action_score)
        loss_macro_action = bi_loss_action(output_macro_action, macro_action_score)

        # 三分类损失
        loss_micro_start_end = cls_loss_func(
            output_micro_start_end.permute(0, 2, 1).contiguous(),
            micro_start_end_label)
        loss_macro_start_end = cls_loss_func(
            output_macro_start_end.permute(0, 2, 1).contiguous(),
            macro_start_end_label)

        # 总损失
        loss = (1.8 * loss_micro_apex
                + 1.0 * loss_micro_start_end
                + 0.1 * loss_micro_action
                + opt['macro_ration'] * (
                    1.0 * loss_macro_apex
                    + 1.0 * loss_macro_start_end
                    + 0.1 * loss_macro_action
                ))

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新损失
        loss_am.update(loss.detach())
        bi_apex_acc_am.update_accuracy(output_micro_apex, micro_apex_score)
        bi_action_acc_am.update_accuracy(output_micro_action, micro_action_score)
        micro_se_acc_am.update_accuracy(output_micro_start_end.permute(0, 2, 1), micro_start_end_label)
        macro_se_acc_am.update_accuracy(output_macro_start_end.permute(0, 2, 1), macro_start_end_label)

        writer.add_scalar("Loss/train", loss, epoch)

        # Log accuracy to TensorBoard
        writer.add_scalar("Accuracy/Micro_Apex", bi_apex_acc_am.accuracy(), epoch)
        writer.add_scalar("Accuracy/Micro_Action", bi_action_acc_am.accuracy(), epoch)
        writer.add_scalar("Accuracy/Micro_Start_End", micro_se_acc_am.accuracy(), epoch)
        writer.add_scalar("Accuracy/Macro_Start_End", macro_se_acc_am.accuracy(), epoch)

    current_lr = optimizer.param_groups[0]['lr']
    results = "[Epoch {0:03d}/{1:03d}]\tLoss {2:.5f}(train)\tAccuracy Micro Apex {3:.2f}%\tAccuracy Micro Action {4:.2f}%\tAccuracy Micro Start-End {5:.2f}%\tAccuracy Macro Start-End {6:.2f}%\tCurrent Learning rate {7:.5f}\n".format(
        epoch, opt["epochs"], loss_am.avg(),
        bi_apex_acc_am.accuracy() * 100,
        bi_action_acc_am.accuracy() * 100,
        micro_se_acc_am.accuracy() * 100,
        macro_se_acc_am.accuracy() * 100,
        current_lr)

    print(results)


    state = {'epoch': epoch + 1,
             'state_dict': model.state_dict()}
    
    ckpt_dir = opt["model_save_root"]
    
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    weight_file = os.path.join(
                    ckpt_dir, 
                    "checkpoint_epoch_" + str(epoch).zfill(3) + ".pth.tar")
    
    # save state_dict every x epochs to save memory
    if (epoch + 1) % opt['save_intervals'] == 0:
        torch.save(state, weight_file)
    print("weight file save in {0}/checkpoint_epoch_{1}.pth.tar\n".format(ckpt_dir, str(epoch).zfill(3)))

            
if __name__ == '__main__':
    from pprint import pprint
    import opts
    
    args = opts.parse_args()
    
    # prep output folder
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    
    
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
    opt['output_dir_name'] = os.path.join(args.output, subject)
    opt['model_save_root'] = os.path.join(opt['output_dir_name'], 'models')
    
    # tensorboard writer
    writer_dir = os.path.join(opt['output_dir_name'], 'logs')
    if not os.path.exists(writer_dir):
        os.makedirs(writer_dir)
    tb_writer = SummaryWriter(writer_dir)
    
    
    # save the current config
    with open(os.path.join(writer_dir, 'config.txt'), 'w') as fid:
        pprint(opt, stream=fid)
        fid.flush()
        
    # prep model
    device = opt['device'] if torch.cuda.is_available() else 'cpu'
    model = AUwGCN(opt)
    model = model.to(device)
    print("Starting training...\n")
    print("Using GPU: {} \n".format(device))
    
    
    # define dataset and dataloader
    train_dataset = LOSO_DATASET(opt, "train", subject)
    # 训练数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt['batch_size'],
                                               shuffle=True,
                                               num_workers=opt['num_workers'])
    
    # # define optimizer and scheduler
    optimizer = configure_optimizers(model, opt["abfcm_training_lr"],
                                     opt["abfcm_weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, opt['abfcm_lr_scheduler'])


    
    for epoch in range(opt['epochs']):
        train(opt, train_loader, model, optimizer, epoch, device, tb_writer)
        scheduler.step()
    
    tb_writer.close()
    print("Finish training!\n")

