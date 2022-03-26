import os
import time
import logging
import itertools
import math
import numpy as np
import random
from PIL import Image
import importlib
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import importlib
import sys
sys.path.append("..")
from utils import util, calculate_PSNR_SSIM
from utils.flowlib import save_flow_image
from utils.pytorch_msssim import ssim_matlab
from models.modules import define_G
from models.losses import PerceptualLoss, AdversarialLoss, EPE, Ternary
from dataloader import DistIterSampler, create_dataloader


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.augmentation = args.data_augmentation
        self.device = torch.device('cuda' if len(args.gpu_ids) != 0 else 'cpu')
        args.device = self.device


        ## init dataloader
        if args.phase == 'train':
            trainset_ = getattr(importlib.import_module('dataloader.dataset'), args.trainset, None)
            self.train_dataset = trainset_(self.args)
            if args.dist:
                dataset_ratio = 1
                train_sampler = DistIterSampler(self.train_dataset, args.world_size, args.rank, dataset_ratio)
                self.train_dataloader = create_dataloader(self.train_dataset, args, train_sampler)
            else:
                self.train_dataloader = DataLoader(self.train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

            self.args.step_per_epoch = self.train_dataloader.__len__()

        else:
            testset_ = getattr(importlib.import_module('dataloader.dataset'), args.testset, None)
            self.test_dataset = testset_(self.args)
            self.test_dataloader = DataLoader(self.test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False)

        ## init network
        self.net = define_G(args)
        if args.resume:
            self.load_networks('net', self.args.resume)

        if args.rank <= 0:
            logging.info('----- generator parameters: %f -----' % (sum(param.numel() for param in self.net.parameters()) / (10**6)))

        ## init loss and optimizer
        if args.phase == 'train':
            if args.rank <= 0:
                logging.info('init criterion and optimizer...')
            g_params = [self.net.parameters()]

            # self.optimizer_G = torch.optim.Adam(itertools.chain.from_iterable(g_params), lr=args.lr, weight_decay=args.weight_decay)
            # self.scheduler = CosineAnnealingLR(self.optimizer_G, T_max=500)  # T_max=args.max_iter
            self.optimizer_G = torch.optim.AdamW(itertools.chain.from_iterable(g_params), lr=args.lr, weight_decay=args.weight_decay)


            if args.loss_l1:
                self.criterion_l1 = nn.L1Loss().to(self.device)
                self.lambda_l1 = args.lambda_l1
                if args.rank <= 0:
                    logging.info('  using l1 loss...')

            if args.loss_flow:
                self.criterion_flow = EPE().to(self.device)
                self.lambda_flow = args.lambda_flow
                if args.rank <= 0:
                    logging.info('  using flow loss...')

            if args.loss_ter:
                self.criterion_ter = Ternary(self.device).to(self.device)
                self.lambda_ter = args.lambda_ter
                if args.rank <= 0:
                    logging.info('  using ter loss...')


            if args.loss_adv:
                self.criterion_adv = AdversarialLoss(gpu_ids=args.gpu_ids, dist=args.dist, gan_type=args.gan_type,
                                                             gan_k=1, lr_dis=args.lr_D, train_crop_size=40)
                self.lambda_adv = args.lambda_adv
                if args.rank <= 0:
                    logging.info('  using adv loss...')

            if args.loss_perceptual:
                self.criterion_perceptual = PerceptualLoss(layer_weights={'conv5_4': 1.}).to(self.device)
                self.lambda_perceptual = args.lambda_perceptual
                if args.rank <= 0:
                    logging.info('  using perceptual loss...')

            if args.resume_optim:
                self.load_networks('optimizer_G', self.args.resume_optim)
            if args.resume_scheduler:
                self.load_networks('scheduler', self.args.resume_scheduler)

    def get_learning_rate(self, step):
        if step < 2000:
            mul = step / 2000.
        else:
            mul = np.cos((step - 2000) / (self.args.max_iter * self.args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return self.args.lr * mul

    def vis_results(self, epoch, i, images):
        for j in range(min(images[0].size(0), 5)):
            save_name = os.path.join(self.args.vis_save_dir, 'vis_%d_%d_%d.jpg' % (epoch, i, j))
            temps = []
            for imgs in images:
                temps.append(imgs[j])
            temps = torch.stack(temps)
            B = temps[:, 0:1, :, :]
            G = temps[:, 1:2, :, :]
            R = temps[:, 2:3, :, :]
            temps = torch.cat([R, G, B], dim=1)
            torchvision.utils.save_image(temps, save_name)

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def prepare(self, batch_samples):
        for key in batch_samples.keys():
            if 'folder' not in key and 'pad_nums' not in key:
                batch_samples[key] = Variable(batch_samples[key].to(self.device), requires_grad=False)

        return batch_samples

    def train(self):
        if self.args.rank <= 0:
            logging.info('training on  ...' + self.args.trainset)
            logging.info('%d training samples' % (self.train_dataset.__len__()))
            logging.info('the init lr: %f'%(self.args.lr))
        steps = self.args.start_iter * self.args.step_per_epoch
        self.net.train()

        if self.args.use_tb_logger:
            if self.args.rank <= 0:
                tb_logger = SummaryWriter(log_dir='tb_logger/' + self.args.name)

        self.best_psnr = 0

        ##################
        # TODO
        # self.args.loss_flow = False
        ##################


        for i in range(self.args.start_iter, self.args.max_iter):
            # self.scheduler.step()
            # logging.info('current_lr: %f' % (self.optimizer_G.param_groups[0]['lr']))

            t0 = time.time()
            for j, batch_samples in enumerate(self.train_dataloader):
                log_info = 'epoch:%03d step:%04d  ' % (i, j)

                # set learning rate
                learning_rate = self.get_learning_rate(steps)
                for param_group in self.optimizer_G.param_groups:
                    param_group['lr'] = learning_rate
                log_info += 'current_lr: %f  ' % (self.optimizer_G.param_groups[0]['lr'])

                ## prepare data
                batch_samples = self.prepare(batch_samples)
                img0 = batch_samples['img0']
                img1 = batch_samples['img1']
                gt = batch_samples['gt']
                flow_gt = batch_samples['flow_gt']

                ## forward
                if not self.args.loss_l1:
                    _, flow_list = self.net(torch.cat([img0, img1], 1))
                else:
                    output, flow_list = self.net(img0, img1, None)

                ## optimization
                loss = 0
                self.optimizer_G.zero_grad()

                if self.args.loss_l1:
                    l1_loss = self.criterion_l1(output, gt)
                    l1_loss = l1_loss * self.lambda_l1
                    loss += l1_loss
                    log_info += 'l1_loss:%.06f ' % (l1_loss.item())

                if self.args.loss_flow:
                    flow_loss = 0
                    for level in range(len(flow_list)):
                        fscale = flow_list[level].size(-1) / flow_gt.size(-1)
                        flow_gt_resize = F.interpolate(flow_gt, scale_factor=fscale, mode="bilinear",
                                             align_corners=False) * fscale
                        flow_loss += self.criterion_flow(flow_list[level][:, :2], flow_gt_resize[:, :2], 1).mean()
                        flow_loss += self.criterion_flow(flow_list[level][:, 2:4], flow_gt_resize[:, 2:4], 1).mean()
                    flow_loss = flow_loss / 2. * self.lambda_flow
                    loss += flow_loss
                    log_info += 'flow_loss:%.06f ' % (flow_loss.item())

                if self.args.loss_ter:
                    ter_loss = self.criterion_ter(output, gt)
                    ter_loss = ter_loss.mean() * self.lambda_ter
                    loss += ter_loss
                    log_info += 'ter_loss:%.06f ' % (ter_loss.item())


                if self.args.loss_perceptual:
                    perceptual_loss, _ = self.criterion_perceptual(output, gt)
                    perceptual_loss = perceptual_loss * self.lambda_perceptual
                    loss += perceptual_loss
                    log_info += 'perceptual_loss:%.06f ' % (perceptual_loss.item())

                if self.args.loss_adv:
                    adv_loss, d_loss = self.criterion_adv(output, gt)
                    adv_loss = adv_loss * self.lambda_adv
                    loss += adv_loss
                    log_info += 'adv_loss:%.06f ' % (adv_loss.item())
                    log_info += 'd_loss:%.06f ' % (d_loss.item())

                log_info += 'loss_sum:%f ' % (loss.item())
                loss.backward()
                self.optimizer_G.step()

                ## print information
                if j % self.args.log_freq == 0:
                    t1 = time.time()
                    log_info += '%4.6fs/batch' % ((t1-t0)/self.args.log_freq)
                    if self.args.rank <= 0:
                        logging.info(log_info)
                    t0 = time.time()

                ## visualization
                if j % self.args.vis_freq == 0:
                    if self.args.loss_l1:
                        vis_temps = [img0, gt, output]
                        self.vis_results(i, j, vis_temps)

                ## write tb_logger
                if self.args.use_tb_logger:
                    if steps % self.args.vis_step_freq == 0:
                        if self.args.rank <= 0:
                            if self.args.loss_l1:
                                tb_logger.add_scalar('l1_loss', l1_loss.item(), steps)
                            if self.args.loss_ter:
                                tb_logger.add_scalar('ter_loss', ter_loss.item(), steps)
                            if self.args.loss_flow:
                                tb_logger.add_scalar('flow_loss', flow_loss.item(), steps)
                            if self.args.loss_perceptual:
                                if i > 5:
                                    tb_logger.add_scalar('perceptual_loss', perceptual_loss.item(), steps)
                            if self.args.loss_adv:
                                if i > 5:
                                    tb_logger.add_scalar('adv_loss', adv_loss.item(), steps)
                                    tb_logger.add_scalar('d_loss', d_loss.item(), steps)

                steps += 1

            ## save networks
            if i % self.args.save_epoch_freq == 0:
                if self.args.rank <= 0:
                    logging.info('Saving state, epoch: %d iter:%d' % (i, 0))
                    self.save_networks('net', i)
                    self.save_networks('optimizer_G', i)
                    # self.save_networks('scheduler', i)

        ## end of training
        if self.args.rank <= 0:
            tb_logger.close()
            self.save_networks('net', 'final')
            logging.info('The training stage on %s is over!!!' % (self.args.trainset))


    def test(self):
        save_path = os.path.join(self.args.save_folder, 'output_imgs')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        self.net.eval()
        logging.info('start testing...')
        logging.info('%d testing samples' % (self.test_dataset.__len__()))
        num = 0

        to_y = False
        PSNR = []
        SSIM = []
        IE_list = []
        with torch.no_grad():
            for batch, batch_samples in enumerate(self.test_dataloader):
                # prepare data
                batch_samples = self.prepare(batch_samples)
                img0 = batch_samples['img0']
                img1 = batch_samples['img1']
                gt = batch_samples['gt']
                folder = batch_samples['folder'][0]
                pad_nums = batch_samples['pad_nums']
                pad_t, pad_d, pad_l, pad_r = pad_nums

                # inference
                output, flow = self.net(img0, img1, None)
                if pad_t != 0 or pad_d != 0 or pad_l != 0 or pad_r != 0:
                    _, _, h, w = output.size()
                    output = output[:, :, pad_t:h-pad_d, pad_l:w-pad_r]

                # calc psnr and ssim
                ssim = ssim_matlab(gt / 255., torch.round(output[0] * 255).unsqueeze(0) / 255.).detach().cpu().numpy()
                mid = np.round((output[0] * 255).detach().cpu().numpy()).astype('uint8').transpose(1, 2, 0) / 255.
                I1 = (gt[0]).detach().cpu().numpy().astype('uint8').transpose(1, 2, 0) / 255.
                psnr = -10 * math.log10(((I1 - mid) * (I1 - mid)).mean())

                PSNR.append(psnr)
                SSIM.append(ssim)
                logging.info('testing on: %s    psnr: %.6f    ssim: %.6f' % (folder, psnr, ssim))

                if self.args.testset == 'MiddleburyDataset':
                    out = np.round(output[0].detach().cpu().numpy().transpose(1, 2, 0) * 255)
                    IE = np.abs((out - (gt[0]).detach().cpu().numpy().astype('uint8').transpose(1, 2, 0) * 1.0)).mean()
                    IE_list.append(IE)
                    logging.info('IE: %.6f' % (IE))

                if self.args.save_result:
                    path = os.path.join(save_path, folder)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    imt = output[0].flip(dims=(0,)).clamp(0., 1.)
                    im0 = img0[0].flip(dims=(0,)).clamp(0., 1.)
                    im1 = img1[0].flip(dims=(0,)).clamp(0., 1.)
                    torchvision.utils.save_image(imt, os.path.join(path, 'imt.png'))
                    torchvision.utils.save_image(im0, os.path.join(path, 'im0.png'))
                    torchvision.utils.save_image(im1, os.path.join(path, 'im1.png'))


                    # flow = flow[0].permute(1, 2, 0).detach().cpu().numpy()
                    # flow_gt = flow_gt[0].permute(1, 2, 0).detach().cpu().numpy()
                    # save_flow_image(flow[:, :, :2], os.path.join(path, 'flow0.png'))
                    # save_flow_image(flow[:, :, 2:], os.path.join(path, 'flow1.png'))
                    # save_flow_image(flow_gt[:, :, :2], os.path.join(path, 'flow0_gt.png'))
                    # save_flow_image(flow_gt[:, :, 2:], os.path.join(path, 'flow1_gt.png'))

                num += 1


        PSNR = np.mean(PSNR)
        SSIM = np.mean(SSIM)
        logging.info('--------- average PSNR: %.06f,  SSIM: %.06f' % (PSNR, SSIM))
        if self.args.testset == 'MiddleburyDataset':
            logging.info('--------- average IE: %.06f' % (np.mean(IE_list)))


    def save_image(self, tensor, path):
        img = Image.fromarray(((tensor/2.0 + 0.5).data.cpu().numpy()*255).transpose((1, 2, 0)).astype(np.uint8))
        img.save(path)

    def load_networks(self, net_name, resume, strict=True):
        load_path = resume
        network = getattr(self, net_name)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        load_net = torch.load(load_path, map_location=torch.device(self.device))
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


    def save_networks(self, net_name, epoch):
        network = getattr(self, net_name)
        save_filename = '{}_{}.pth'.format(net_name, epoch)
        save_path = os.path.join(self.args.snapshot_save_dir, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        if not 'optimizer' and not 'scheduler' in net_name:
            for key, param in state_dict.items():
                state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)
