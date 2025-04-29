from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.type = 'image'
_C.root = '/home/peng/DMU/datasets/'
_C.sources = ['occluded_duke'] #['occluded_duke', market1501, occluded_reid]
_C.targets = ['occluded_duke', 'occluded_duke']
_C.workers = 0 # number of data loading workers, set to 0 to enable breakpoint debugging in dataloader code
_C.split_id = 0 # split index
_C.height = 256 # image height
_C.width = 128 # image width
_C.combineall = False # combine train, query and gallery for training
_C.transforms = ['rc', 're']  # data augmentation from ['rf', 'rc', 're', 'cj'] = ['random flip', 'random crop', 'random erasing', 'color jitter']
_C.norm_mean = [0.485, 0.456, 0.406] # default is imagenet mean
_C.norm_std = [0.229, 0.224, 0.225] # default is imagenet std
_C.save_dir = 'logs'  # save figures, images, logs, etc. in this folder
_C.load_train_targets = False
_C.train_sampler = 'RandomIdentitySampler'  # sampler for source train loader
_C.train_sampler_t = 'RandomIdentitySampler'  # sampler for target train loader
_C.num_instances = 4  # number of instances per identity for RandomIdentitySampler
_C.test_batch_size = 128
_C.train_batch_size = 64
_C.use_gpu = True
# mask
_C.mask_loss = 'part_based'
_C.masks_dir = 'pifpaf_maskrcnn_filtering'
_C.masks = CN()
_C.masks.type = 'disk'  # when 'disk' is used, load part masks from storage in '_C.masks.dir' folder
# when 'stripes' is used, divide the image in '_C.masks.parts_num' horizontal stripes in a PCB style.
# 'stripes' with parts_num=1 can be used to emulate the global method Bag of Tricks (BoT)
_C.masks.parts_num = 8  # number of part-based embedding to extract. When PCB is used, change this parameter to the number of stripes required
_C.masks.dir = 'pifpaf_maskrcnn_filtering'  # masks will be loaded from 'dataset_path/masks/<_C.masks.dir>' directory
_C.masks.preprocess = 'eight'  # how to group the 36 pifpaf parts into smaller human semantic groups ['eight', 'five', 'four', 'two', ...], more combination available inside 'torchreid/data/masks_transforms/__init__.masks_preprocess_pifpaf'
_C.masks.softmax_weight = 15
_C.masks.background_computation_strategy = 'threshold'  # threshold, diff_from_max
_C.masks.mask_filtering_threshold = 0.5
_C.cj = CN()
_C.cj.brightness = 0.3
_C.cj.contrast = 0.3
_C.cj.saturation = 0.2
_C.cj.hue = 0.2
_C.cj.always_apply = False
_C.cj.p = 0.5


