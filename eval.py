import torch
import opts
from model import AUwGCN
from datasets import LOSO_DATASET
import os
import yaml

from utils.eval_utils import eval_single_epoch, nms_single_epoch, calculate_epoch_metrics, \
                             choose_best_epoch

if __name__ == '__main__':
    import opts
    args = opts.parse_args()
    
    # load config & params.
    with open("/kaggle/working/AUW-GCN-test/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        if args.dataset is not None:
            dataset = args.dataset
        else:
            dataset = yaml_config['dataset']
        opt = yaml_config[dataset]
        opt['dataset'] = dataset
        
    subject = args.subject
    
    # update opt. according to args.
    opt['output_dir_name'] = os.path.join(args.output, subject) # ./debug/casme_016
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
    model = AUwGCN(opt)
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
                # 这里为什么是cpu
                checkpoint = torch.load(weight_file,
                                        map_location=torch.device("cpu"))
                model.load_state_dict(checkpoint['state_dict'])
                eval_single_epoch(opt, model, dataloader, epoch, device)

                nms_single_epoch(opt, epoch)

    print("Calculate metrics of all the epochs\n")
    # calculate metrics of all the epochs
    calculate_epoch_metrics(opt)
    print("epoch_metrics csv save in {0}/epoch_metrics.csv\n".format(opt['output_dir_name']))

    print("Choose the best epoch according to criterion\n")
    # choose the best epoch according to criterion
    choose_best_epoch(opt, criterion='all_f1')
    print("best_res csv save in {0}/best_res.csv\n".format(opt['output_dir_name']))

    print("Finish evaluating!\n")