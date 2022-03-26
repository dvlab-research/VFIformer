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
    parser = argparse.ArgumentParser(description='Frame Interpolation Testing')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--name', default='test_vfiformer', type=str)
    parser.add_argument('--phase', default='test', type=str)

    ## device setting
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    ## network setting
    parser.add_argument('--net_name', default='VFIformer', type=str, help='')
    parser.add_argument('--window_size', default=8, type=int)
    parser.add_argument('--module_scale_factor', default=2, type=int)
    parser.add_argument('--input_nc', default=3, type=int)
    parser.add_argument('--output_nc', default=3, type=int)

    ## dataloader setting
    parser.add_argument('--data_root', default='/home/liyinglu/newData/datasets/vfi/SNU-FILM/',type=str)
    parser.add_argument('--testset', default='FILM', type=str, help='FILM')
    parser.add_argument('--test_level', default='extreme', type=str, help='easy|medium|hard|extreme')
    parser.add_argument('--crop_size', default=192, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--data_augmentation', default=False, type=bool)
    
    parser.add_argument('--resume', default='./pretrained_models/pretrained_VFIformer/net_220.pth', type=str)
    parser.add_argument('--resume_flownet', default='', type=str)
    parser.add_argument('--save_folder', default='./test_results', type=str)
    parser.add_argument('--save_result', action='store_true')

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

    #### distributed training settings
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
    # save paths
    save_path = os.path.join(args.save_folder, args.testset)
    log_file_path = save_path + '/' + time.strftime('%Y%m%d_%H%M%S') + '.log'
    save_path = os.path.join(save_path, 'output_imgs')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    setup_logger(log_file_path)

    ## load model
    device = torch.device('cuda' if len(args.gpu_ids) != 0 else 'cpu')
    args.device = device
    net = define_G(args)
    net = load_networks(net, args.resume)

    net.eval()

    level = args.test_level  # levels = ['easy', 'medium', 'hard', 'extreme']
    data_list = []
    with open('%s/test-%s.txt' % (args.data_root, level), 'r') as txt:
        sequence_list = [line.strip() for line in txt]

    for seq in sequence_list:
        img0, gt, img1 = seq.split(' ')
        img0 = os.path.join(args.data_root, img0.replace('data/SNU-FILM/', ''))
        img1 = os.path.join(args.data_root, img1.replace('data/SNU-FILM/', ''))
        gt = os.path.join(args.data_root, gt.replace('data/SNU-FILM/', ''))
        folder = gt
        data_list.append([img0, img1, gt, folder])


    logging.info('--- totol images: %d ---' % (len(data_list)))

    PSNR = []
    SSIM = []
    down_scale = 0.5
    for idx in range(len(data_list)):
        I0, I1, It, folder = data_list[idx]
        img0 = cv2.imread(I0)
        img1 = cv2.imread(I1)
        gt = cv2.imread(It)

        # # pad HR to be mutiple of 64
        # h, w, c = gt.shape
        # if h % 64 != 0 or w % 64 != 0:
        #     h_new = math.ceil(h / 64) * 64
        #     w_new = math.ceil(w / 64) * 64
        #     pad_t = 0
        #     pad_d = h_new - h
        #     pad_l = 0
        #     pad_r = w_new - w
        #     img0 = cv2.copyMakeBorder(img0.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)  # cv2.BORDER_REFLECT
        #     img1 = cv2.copyMakeBorder(img1.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        # else:
        #     pad_t, pad_d, pad_l, pad_r = 0, 0, 0, 0

        # img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)
        # img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)
        # gt = torch.from_numpy(gt).permute(2, 0, 1).cuda().unsqueeze(0)

        # with torch.no_grad():
        #     img0_down = F.interpolate(img0, scale_factor=down_scale, mode="bilinear", align_corners=False)
        #     img1_down = F.interpolate(img1, scale_factor=down_scale, mode="bilinear", align_corners=False)
        #     b, c, h, w = img0_down.size()
        #     if h % 64 != 0 or w % 64 != 0:
        #         h_new = math.ceil(h / 64) * 64
        #         w_new = math.ceil(w / 64) * 64
        #         img0_new = torch.zeros((b, c, h_new, w_new)).to(gt.device).float()
        #         img1_new = torch.zeros((b, c, h_new, w_new)).to(gt.device).float()
        #         img0_new[:, :, :h, :w] = img0_down
        #         img1_new[:, :, :h, :w] = img1_down
        #         img0_down = img0_new
        #         img1_down = img1_new

        #     flow_down = net.get_flow(img0_down, img1_down)
        #     if h % 64 != 0 or w % 64 != 0:
        #         flow_down = flow_down[:, :, :h, :w]

        #     flow = F.interpolate(flow_down, scale_factor=1/down_scale, mode="bilinear", align_corners=False) * 1/down_scale

        #     output = sliding_forward(net, img0, img1, flow, device)

        # if pad_t != 0 or pad_d != 0 or pad_l != 0 or pad_r != 0:
        #     _, _, h, w = output.size()
        #     output = output[:, :, pad_t:h-pad_d, pad_l:w-pad_r]


########################################################
        # pad HR to be mutiple of 64
        h, w, c = gt.shape
        if h % 64 != 0 or w % 64 != 0:
            h_new = math.ceil(h / 64) * 64
            w_new = math.ceil(w / 64) * 64
            pad_t = 0
            pad_d = h_new - h
            pad_l = 0
            pad_r = w_new - w
            img0 = cv2.copyMakeBorder(img0.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)  # cv2.BORDER_REFLECT
            img1 = cv2.copyMakeBorder(img1.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        else:
            pad_t, pad_d, pad_l, pad_r = 0, 0, 0, 0

        img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)
        img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1).cuda().unsqueeze(0)
        gt = torch.from_numpy(gt).permute(2, 0, 1).cuda().unsqueeze(0)

        with torch.no_grad():
            img0_down = F.interpolate(img0, scale_factor=down_scale, mode="bilinear", align_corners=False)
            img1_down = F.interpolate(img1, scale_factor=down_scale, mode="bilinear", align_corners=False)
            b, c, h, w = img0_down.size()
            if h % 64 != 0 or w % 64 != 0:
                h_new = math.ceil(h / 64) * 64
                w_new = math.ceil(w / 64) * 64
                img0_new = torch.zeros((b, c, h_new, w_new)).to(gt.device).float()
                img1_new = torch.zeros((b, c, h_new, w_new)).to(gt.device).float()
                img0_new[:, :, :h, :w] = img0_down
                img1_new[:, :, :h, :w] = img1_down
                img0_down = img0_new
                img1_down = img1_new

            flow_down = net.get_flow(img0_down, img1_down)
            if h % 64 != 0 or w % 64 != 0:
                flow_down = flow_down[:, :, :h, :w]

            flow = F.interpolate(flow_down, scale_factor=1/down_scale, mode="bilinear", align_corners=False) * 1/down_scale

            # output = sliding_forward(net, img0, img1, flow, device)
            output, _,  = net(img0, img1, flow_pre=flow)

        if pad_t != 0 or pad_d != 0 or pad_l != 0 or pad_r != 0:
            _, _, h, w = output.size()
            output = output[:, :, pad_t:h-pad_d, pad_l:w-pad_r]

        ssim = ssim_matlab(gt / 255., torch.round(output[0] * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
        mid = np.round((output[0] * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.
        I1 = (gt[0]).detach().cpu().numpy().astype('uint8').transpose(1, 2, 0) / 255.
        psnr = -10 * math.log10(((I1 - mid) * (I1 - mid)).mean())
        # ssim = 0
        PSNR.append(psnr)
        SSIM.append(ssim)
        logging.info('testing on: %s    psnr: %.6f    ssim: %.6f' % (It, psnr, ssim))

        if args.save_result:
            save_folder = args.save_folder + '_' + args.test_level
            imt = output[0].flip(dims=(0,)).clamp(0., 1.)
            basefoler = It.split('/')
            save_folder = os.path.join(save_folder, basefoler[-3], basefoler[-2])
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            torchvision.utils.save_image(imt, os.path.join(save_folder, os.path.basename(It)))


        logging.info('--------- average PSNR: %.06f,  SSIM: %.06f' % (np.mean(PSNR), np.mean(SSIM)))
        # torch.cuda.empty_cache()


    logging.info('***************************************************')
    PSNR = np.mean(PSNR)
    SSIM = np.mean(SSIM)
    logging.info('--------- average PSNR: %.06f,  SSIM: %.06f' % (PSNR, SSIM))




def sliding_forward(net, img0, img1, flow, device, crop_size=(2000, 2000), stride=(384, 896)): # crop_size=(768, 1280), stride=(384, 896)
    h, w = img0.size()[2:]
    if h <= crop_size[0] and w <= crop_size[1]:
        output, _, _ = net(img0, img1, flow_gt=flow)
        return output

    else:
        result = torch.zeros(1, 3, h, w).cuda()
        count = torch.zeros(1, 1, h, w).cuda()
        h_steps = 1 + int(math.ceil(float(max(h - crop_size[0], 0)) / stride[0]))
        w_steps = 1 + int(math.ceil(float(max(w - crop_size[1], 0)) / stride[1]))

        for h_idx in range(h_steps):
            for w_idx in range(w_steps):
                ws0, ws1 = w_idx * stride[1], crop_size[1] + w_idx * stride[1]
                hs0, hs1 = h_idx * stride[0], crop_size[0] + h_idx * stride[0]
                if h_idx == h_steps - 1:
                    hs0, hs1 = max(h - crop_size[0], 0), h
                if w_idx == w_steps - 1:
                    ws0, ws1 = max(w - crop_size[1], 0), w

                img0_crop = img0[:, :, hs0:hs1, ws0:ws1]
                img1_crop = img1[:, :, hs0:hs1, ws0:ws1]
                flow_crop = flow[:, :, hs0:hs1, ws0:ws1]

                output, _, _ = net(img0_crop, img1_crop, flow_gt=flow_crop)

                result[:, :, hs0: hs1, ws0: ws1] += output
                count[:, :, hs0: hs1, ws0: ws1] += 1

        assert torch.min(count) > 0
        result = result / count
        return result



# def sliding_forward(net, img0, img1, device, crop_size=1280, stride=640): #crop_size=1440, stride=1260
#     h, w = img0.size()[2:]
#     if w <= crop_size:
#         output, flow, merged_img = net(img0, img1, None)
#         return output
#     else:
#         result = torch.zeros(1, 3, h, w).cuda()
#         count = torch.zeros(1, 1, h, w).cuda()
#         w_steps = 1 + int(math.ceil(float(max(w - crop_size, 0)) / stride))
        
#         for w_idx in range(w_steps):
#             ws0, ws1 = w_idx * stride, crop_size + w_idx * stride
#             if w_idx == w_steps - 1:
#                 ws0, ws1 = max(w - crop_size, 0), w

#             img0_crop = img0[:, :, :, ws0:ws1]
#             img1_crop = img1[:, :, :, ws0:ws1]
#             output, flow, merged_img = net(img0_crop, img1_crop, None)
#             result[:, :, :, ws0: ws1] += output
#             count[:, :, :, ws0: ws1] += 1

#         assert torch.min(count) > 0
#         result = result / count
#         return result


if __name__ == '__main__':
    main()


