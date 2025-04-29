from utils.logger import setup_logger
from data import datamanager
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train, do_inference
import random
import torch
import numpy as np
import os
import argparse
# from timm.scheduler import create_scheduler
from config import cfg
from config import dataConfig
from data.masks_transforms import compute_parts_num_and_names
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def imagedata_kwargs(cfg):
    return {
        'config': cfg,
        'root': cfg.root,
        'sources': cfg.sources,
        'targets': cfg.targets,
        'height': cfg.height,
        'width': cfg.width,
        'transforms': cfg.transforms,
        'norm_mean': cfg.norm_mean,
        'norm_std': cfg.norm_std,
        'use_gpu': cfg.use_gpu,
        'split_id': cfg.split_id,
        'combineall': cfg.combineall,
        'load_train_targets': cfg.load_train_targets,
        'batch_size_train': cfg.train_batch_size,
        'batch_size_test': cfg.test_batch_size,
        'workers': cfg.workers,
        'num_instances': cfg.num_instances,
        'train_sampler': cfg.train_sampler,
        'train_sampler_t': cfg.train_sampler_t,
        'masks_dir': cfg.masks_dir,
        'use_masks': True
    }

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    Inform = "xr:-LRF +FAM(s)"
    logger = setup_logger("transreid", output_dir, if_train=True, inform=Inform)

    logger.info(Inform)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    dataCfg = imagedata_kwargs(dataConfig)
    compute_parts_num_and_names(dataConfig)
    dataManager = datamanager.ImageDataManager(**dataCfg)   # source target maskdir
    num_classes = dataManager.num_train_pids
    camera_num = dataManager.num_train_cams
    view_num = dataManager.num_train_vids
    # num_query  occluded_reid : dataManager.test_dataset[dataManager.targets[1]   occluded_duke : dataManager.test_dataset[dataManager.targets[0]
    train_loader, val_loader, num_query = dataManager.train_loader, dataManager.test_loader, len(dataManager.test_dataset[dataManager.targets[1]]['query'])
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    # model.load_param("/home/peng/DMU/transformer_150.pth")
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    optimizer, Moptimizer = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)
    cfg.defrost()
    cfg.MODEL.MASK_NUM  = dataConfig.masks.parts_num
    cfg.DATASETS.NAMES = dataConfig.targets
    cfg.freeze()
    # do_inference(cfg,
    #              model,
    #              val_loader,
    #              num_query)

    # from thop import  profile
    # flops,params = profile(model, inputs = (torch.randn(64, 3, 256, 128),))
    # print("FLOPS: %.2fG" %(flops / 1e9))
    # print(flops)
    # print("Params: %.2fM" %(params / 1e6))
    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        Moptimizer,
        scheduler,
        loss_func,
        num_query, args.local_rank
    )
