import os.path as osp
import numpy as np
from easydict import EasyDict
import torch


cfg = EasyDict()

cfg.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cfg.NUM_WORKERS = 4
cfg.IMG_MEAN = np.array((90.0, 98.0, 102.0), dtype=np.float32)
cfg.NUM_FRAMES = 40
cfg.CROP_SIZE = (224, 224)

cfg.BATCH_SIZE_TRAIN = 16
cfg.LR_FIRST_UDA = 5e-4
cfg.LR_SECOND_UDA = 1e-4
cfg.EPOCH_FIRST_UDA = 100
cfg.EPOCH_SECOND_UDA = 60
cfg.THETA = 0.8
cfg.TRAIN_SHUFFLE = True
cfg.MIN_LOSS = 100

cfg.SOURCE_TXT_DIR = 'source.txt'
cfg.SOURCE_LABEL_DIR = 'source.xls'
cfg.SOURCE_FRAME_DIR = 'source_frame\\'
cfg.SOURCE_DATA_SAMPLE = 1000

cfg.TARGET_TXT_DIR = 'target.txt'
cfg.TARGET_LABEL_DIR = 'target.xls'
cfg.TARGET_FRAME_DIR = 'target_frame\\'
cfg.TARGET_DATA_SAMPLE = 1000

cfg.BATCH_SIZE_TEST = 1
cfg.ETA = 0.6
cfg.EPSILON = 0.5

cfg.BEST_MODEL_DIR = 'best_model.pth'
cfg.CONFIDENT_TXT_DIR = 'confident_split.txt'
cfg.UNCERTAIN_TXT_DIR = 'uncertain_split.txt'
cfg.CONFIDENT_LABEL_DIR = 'pseudo_target.xls'
cfg.UNCERTAIN_LABEL_DIR = 'pseudo_target.xls'
