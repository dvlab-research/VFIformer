import os
import sys
import time
import logging
import math
import glob
import cv2
import argparse
import numpy as np
from torch.nn.parallel import DataParallel, DistributedDataParallel
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.utils.data as data
from skimage.color import rgb2yuv, yuv2rgb

from utils.util import setup_logger, print_args
from utils.pytorch_msssim import ssim_matlab
from models import modules
from models.modules import define_G



def load_networks(network, resume, strict=True):
    load_path = resume
    if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
        network = network.module
    load_net = torch.load(load_path, map_location=torch.device('cpu'))
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    if 'optimizer' or 'scheduler' in net_name:
        network.load_state_dict(load_net_clean)
    else:
        network.load_state_dict(load_net_clean, strict=strict)

    return network



def main():
    parser = argparse.ArgumentParser(description='inference for a single sample')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--name', default='demo', type=str)
    parser.add_argument('--phase', default='test', type=str)

    ## device setting
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    ## network setting
    parser.add_argument('--net_name', default='VFIformer', type=str, help='')

    ## dataloader setting
    parser.add_argument('--crop_size', default=192, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    
    # parser.add_argument('--img0_path', type=str, required=True)
    # parser.add_argument('--img1_path', type=str, required=True)

    # external deps
    parser.add_argument('--resume', default='./pretrained_models/pretrained_VFIformer/net_220.pth', type=str)
    parser.add_argument('--resume_flownet', default='', type=str)

    parser.add_argument('--save_folder', default='./output', type=str)

    parser.add_argument('--base_path', default='./images', type=str, help="path to png files")
    parser.add_argument('--base_name', default='image', type=str, help="filename before 0-filled index number")
    parser.add_argument('--img_first', default=0, type=int, help="first image index, usually 0")
    parser.add_argument('--img_last', default=2, type=int, help="last image index, usually count-1")
    parser.add_argument('--img_step', default=1, type=int, help="step through the image indexes, should be 1")
    # parser.add_argument('--img_offset', default=0, type=int, help="0=odd, 1=even frame generation")

    ## setup training environment
    args = parser.parse_args()

    ## setup training device
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

    ## distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        args.dist = False
        args.rank = -1
        print('Disabled distributed training.')
    else:
        args.dist = True
        init_dist()
        args.world_size = torch.distributed.get_world_size()
        args.rank = torch.distributed.get_rank()


    cudnn.benchmark = True
    ## save paths
    save_path = args.save_folder

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ## load model
    device = torch.device('cuda' if len(args.gpu_ids) != 0 else 'cpu')
    args.device = device
    net = define_G(args)
    net = load_networks(net, args.resume)
    net.eval()

    ## load data


    divisor = 64
    multi = 3

    basepath = args.base_path #"C:\CONTENT\BABY PICTURES\BABY JERRY AND AUNTS CC"
    basefile = args.base_name #"BABY JERRY AND AUNTS"
    start = args.img_first #0
    end = args.img_last #2674
    step = args.img_step #2
    num_width = len(str(end)) 

    for n in range(start, end, step):
      file_a = basepath + "\\" + basefile + str(n).zfill(num_width) + ".png"
      file_b = basepath + "\\" + basefile + str(n + step).zfill(num_width) + ".png"

      translated_index = n * 2
      
      file_at = os.path.join(save_path, basefile + str(translated_index).zfill(num_width) + ".png")
      file_bt = os.path.join(save_path, basefile + str(translated_index + 2).zfill(num_width) + ".png")
      file_ab = os.path.join(save_path, basefile + str(translated_index + 1).zfill(num_width) + ".png")

      file0 = img0 = cv2.imread(file_a)
      file1 = img1 = cv2.imread(file_b)

      h, w, c = img0.shape
      if h % divisor != 0 or w % divisor != 0:
          h_new = math.ceil(h / divisor) * divisor
          w_new = math.ceil(w / divisor) * divisor
          pad_t = (h_new - h) // 2
          pad_d = (h_new - h) // 2 + (h_new - h) % 2
          pad_l = (w_new - w) // 2
          pad_r = (w_new - w) // 2 + (w_new - w) % 2
          img0 = cv2.copyMakeBorder(img0.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)  # cv2.BORDER_REFLECT
          img1 = cv2.copyMakeBorder(img1.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
      else:
          pad_t, pad_d, pad_l, pad_r = 0, 0, 0, 0

      img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)
      img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)

      with torch.no_grad():
          output, _ = net(img0, img1, None)
          h, w = output.size()[2:]
          output = output[:, :, pad_t:h-pad_d, pad_l:w-pad_r]

      imt = output[0].flip(dims=(0,)).clamp(0., 1.)
      cv2.imwrite(file_at, file0)
      print("saved pre " + file_at)

      torchvision.utils.save_image(imt, file_ab)
      print("saved int " + file_ab)

      if n >= (end - step):
        cv2.imwrite(file_bt, file1)
        print("saved pst " + file_bt)

if __name__ == '__main__':
    main()


