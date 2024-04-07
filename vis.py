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
import seaborn as sns
from einops import rearrange

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


def corr_compute(feat):
    Q = rearrange(feat, 'n c h w -> n c (h w)')
    K = rearrange(feat, 'n c h w -> n c (h w)')
    corr_map = torch.matmul(Q.transpose(1, 2), K) / torch.sqrt(torch.tensor(Q.shape[1]).float())
    corr_map = F.softmax(corr_map, dim=-1)
    return corr_map


def corr2heatmap_save(corr_map_i, pixel_index):
    temp_map = rearrange(corr_map_i[pixel_index], '(h w) -> 1 1 h w', h=c4_feats_i.shape[-2], w=c4_feats_i.shape[-1])
    temp_map = F.interpolate(temp_map, (h, w), mode='bilinear')
    temp_map = rearrange(temp_map, '1 1 h w -> h w')
    range_ = torch.max(temp_map) - torch.min(temp_map)
    temp_map = (- torch.min(temp_map) + temp_map) / range_
    plt.figure(figsize=(w / 50, h / 50))
    heat_map = sns.heatmap(temp_map.cpu().numpy(), cbar=False)
    heat_map = heat_map.get_figure()
    plt.axis('off')

    heat_map.savefig('temp/{}/{}_{}_corr.png'.format(file_name, int(pixel_index / c4_feats_i.shape[-1]) * int(
        h / c4_feats_i.shape[-2]), int(pixel_index % c4_feats_i.shape[-1]) * int(w / c4_feats_i.shape[-1])),
                     pad_inches=0, bbox_inches='tight')
    plt.clf()
    plt.close()
    del heat_map, temp_map


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

with torch.no_grad():
    for img, mask, ids, img_ori in valloader:
        dist.barrier()
        img = img.cuda()
        b, _, h, w = img.shape
        res = model(img, use_corr=True)
        pred = res['out']
        pred_mask = pred.argmax(dim=1)
        pred_conf = pred.softmax(dim=1).max(dim=1)[0]
        pred_conf_fliter = (pred_conf <= 0.95)
        mask_fliter = pred_mask.clone()
        mask_fliter[pred_conf_fliter] = 255
        corr_map = res['corr_map']
        c4_feats = res['c4']

        for i in range(pred_mask.shape[0]):
            file_name = ids[i].split(' ')[0].split('/')[1].split('.')[0]
            if not os.path.exists('temp/{}'.format(file_name)):
                os.mkdir('temp/{}'.format(file_name))
            print(file_name)
            mask_pred_i = pred_mask[i]
            mask_i = mask[i]
            mask_filter_i = mask_fliter[i]
            corr_map_i = corr_map[i]
            c4_feats_i = c4_feats[i]
            for pixel_index in range(corr_map_i.shape[0]):
                corr2heatmap_save(corr_map_i, pixel_index)
