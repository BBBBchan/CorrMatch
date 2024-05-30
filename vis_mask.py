import argparse
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from einops import rearrange
import numpy as np
from PIL import Image

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus_vis import DeepLabV3Plus
from util.dist_helper import setup_distributed

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
args = parser.parse_args()

def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

    return cmap


rank, word_size = setup_distributed(port=args.port)

cfg = yaml.load(open('configs/pascal.yaml', "r"), Loader=yaml.Loader)

model = DeepLabV3Plus(cfg)
model.load_state_dict(torch.load('Your/checkpoint/path'))
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model.cuda()

local_rank = int(os.environ["LOCAL_RANK"])
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                  output_device=local_rank, find_unused_parameters=False)

valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
valsampler = torch.utils.data.distributed.DistributedSampler(valset, shuffle=False)
valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4,
                       drop_last=False, sampler=valsampler)

model.eval()
if local_rank == 0:
    if not os.path.exists('visual'):
        os.mkdir('visual')

with torch.no_grad():
    for img, mask, ids, img_ori in valloader:
        dist.barrier()

        img = img.cuda()
        b, _, h, w = img.shape
        res = model(img, use_corr=True)
        pred = res['out']
        pred_mask = pred.argmax(dim=1)
        pred_conf = pred.softmax(dim=1).max(dim=1)[0]
        # take 0.95 as an example
        pred_conf_fliter = (pred_conf <= 0.95)
        mask_fliter = pred_mask.clone()
        mask_fliter[pred_conf_fliter] = 255
        for i in range(pred_mask.shape[0]):
            file_name = ids[i].split(' ')[0].split('/')[1].split('.')[0]
            if not os.path.exists('visual/{}'.format(file_name)):
                os.mkdir('visual/{}'.format(file_name))
            print(file_name)
            mask_pred_i = pred_mask[i]
            mask_i = mask[i]
            mask_filter_i = mask_fliter[i]
            mask_i = Image.fromarray(mask_i.cpu().numpy().astype(np.uint8), mode='P')
            mask_pred_i = Image.fromarray(mask_pred_i.cpu().numpy().astype(np.uint8), mode='P')
            mask_filter_i = Image.fromarray(mask_filter_i.cpu().numpy().astype(np.uint8), mode='P')
            platte = color_map()
            mask_i.putpalette(platte)
            mask_pred_i.putpalette(platte)
            mask_filter_i.putpalette(platte)
            mask_i.save('visual/{}/mask_gt.png'.format(file_name))
            mask_pred_i.save('visual/{}/mask_pred.png'.format(file_name))
            mask_filter_i.save('visual/{}/mask_filter.png'.format(file_name))

