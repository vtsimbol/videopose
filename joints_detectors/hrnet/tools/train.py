# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import joints_detectors.hrnet.tools._init_paths
from joints_detectors.hrnet.lib.config import cfg
from joints_detectors.hrnet.lib.config import update_config
from joints_detectors.hrnet.lib.core.loss import JointsMSELoss
from joints_detectors.hrnet.lib.core.function import train
from joints_detectors.hrnet.lib.core.function import validate
import joints_detectors.hrnet.lib.dataset as dataset
from joints_detectors.hrnet.lib.utils.utils import get_optimizer
from joints_detectors.hrnet.lib.utils.utils import save_checkpoint
from joints_detectors.hrnet.lib.utils.utils import create_logger
from joints_detectors.hrnet.lib.utils.utils import get_model_summary
from joints_detectors.hrnet.lib.utils.custom_torch_transforms import NormalizeEachImg


import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def get_state_dict(model):
    try:
        state_dict = model.module.state_dict()
    except AttributeError:
        state_dict = model.state_dict()
    return state_dict


def copy_prev_models(prev_models_dir, model_dir):
    import shutil

    vc_folder = '/hdfs/' \
        + '/' + os.environ['PHILLY_VC']
    source = prev_models_dir
    # If path is set as "sys/jobs/application_1533861538020_2366/models" prefix with the location of vc folder
    source = vc_folder + '/' + source if not source.startswith(vc_folder) \
        else source
    destination = model_dir

    if os.path.exists(source) and os.path.exists(destination):
        for file in os.listdir(source):
            source_file = os.path.join(source, file)
            destination_file = os.path.join(destination, file)
            if not os.path.exists(destination_file):
                print("=> copying {0} to {1}".format(
                    source_file, destination_file))
            shutil.copytree(source_file, destination_file)
    else:
        print('=> {} or {} does not exist'.format(source, destination))


def main():
    args = parse_args()
    update_config(cfg, args)

    if args.prevModelDir and args.modelDir:
        # copy pre models for philly
        copy_prev_models(args.prevModelDir, args.modelDir)

    logger, final_output_dir, tb_log_dir = create_logger(cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    ckpt = torch.load(cfg.MODEL.PRETRAINED) if cfg.MODEL.PRETRAINED != '' else None
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=True, ckpt=ckpt)

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'), final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand((1, 1 if cfg.MODEL.GRAYSCALE else 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0]))
    # writer_dict['writer'].add_graph(model, (dump_input, ))

    logger.info(get_model_summary(model, dump_input))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()

    # Data loading code
    if cfg.MODEL.NORM_EACH_IMG:
        normalize = NormalizeEachImg()
    else:
        normalize = transforms.Normalize(mean=[0.449], std=[0.226]) if cfg.MODEL.GRAYSCALE else \
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = []
    valid_dataset = []
    for i in range(len(cfg.DATASET.DATASET)):
        if cfg.DATASET.TRAIN_SET[i] != '':
            train_dataset.append(eval('dataset.' + cfg.DATASET.DATASET[i])(cfg, cfg.DATASET.ROOT[i],
                                                                           cfg.DATASET.TRAIN_SET[i], True, transf))
        if cfg.DATASET.TEST_SET[i] != '':
            valid_dataset.append(eval('dataset.' + cfg.DATASET.DATASET[i])(cfg, cfg.DATASET.ROOT[i],
                                                                           cfg.DATASET.TEST_SET[i], False, transf))
        else:
            valid_dataset.append(None)

    train_dataset = dataset.Concatenator(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = []
    for ds in valid_dataset:
        if ds is not None:
            valid_loader.append(
                torch.utils.data.DataLoader(ds, batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS), shuffle=False,
                                            num_workers=cfg.WORKERS, pin_memory=cfg.PIN_MEMORY)
            )

    best_ap = 0.0
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    if cfg.MODEL.PRETRAINED != '':
        optimizer.load_state_dict(ckpt['optimizer'])
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH if cfg.MODEL.PRETRAINED == '' else ckpt['epoch']
    checkpoint_file = os.path.join(final_output_dir, 'checkpoint.pth')

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_ap = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {checkpoint["epoch"]})')

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.TRAIN.LR_STEP,
                                                        cfg.TRAIN.LR_FACTOR, last_epoch=last_epoch)

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        train(cfg, train_loader, model, criterion, optimizer, epoch, final_output_dir, tb_log_dir, writer_dict)

        score = []
        for i in range(len(cfg.DATASET.DATASET)):
            if valid_dataset[i] is not None:
                dataset_name = os.path.basename(cfg.DATASET.ROOT[i])
                logger.info(f'{dataset_name} validation')
                score.append(
                    validate(cfg, valid_loader[i], valid_dataset[i], model, criterion, final_output_dir,
                             tb_log_dir, writer_dict, root_name=dataset_name)
                )

        score = min(score)
        lr_scheduler.step()
        logger.info(f'Epoch {epoch} | current learning rate: {lr_scheduler.get_lr()} | main val score (AP): {score:.3f}')

        if score >= best_ap:
            best_ap = score
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch,
            'model': cfg.MODEL.NAME,
            'state_dict': get_state_dict(model),
            'perf': score,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
