import os
import sys
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
import functools
import copy
from functools import partial, reduce
import numpy as np
import itertools
import math
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
sys.path.append('../..')
from models.archs.warplayer import warp
from models.archs.transformer_layers import TFModel


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, nf, kernel_size=3, stride=1, padding=1, dilation=1, act='relu'):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        out = self.conv2(self.act(self.conv1(x)))

        return out + x


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.Sequential(
        torch.nn.ConvTranspose2d(in_channels=in_planes, out_channels=out_planes, kernel_size=4, stride=2, padding=1),
        nn.PReLU(out_planes)
    )


def conv_wo_act(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        )


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class Conv2(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2):
        super(Conv2, self).__init__()
        self.conv1 = conv(in_planes, out_planes, 3, stride, 1)
        self.conv2 = conv(out_planes, out_planes, 3, 1, 1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class IFBlock(nn.Module):
    def __init__(self, in_planes, scale=1, c=64):
        super(IFBlock, self).__init__()
        self.scale = scale
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
            conv(c, c),
        )
        self.conv1 = nn.ConvTranspose2d(c, 4, 4, 2, 1)

    def forward(self, x):
        if self.scale != 1:
            x = F.interpolate(x, scale_factor= 1. / self.scale, mode="bilinear", align_corners=False)
        x = self.conv0(x)
        x = self.convblock(x) + x
        x = self.conv1(x)
        flow = x
        if self.scale != 1:
            flow = F.interpolate(flow, scale_factor= self.scale, mode="bilinear", align_corners=False)
        return flow

    
class IFNet(nn.Module):
    def __init__(self, args=None):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(6, scale=4, c=240)
        self.block1 = IFBlock(10, scale=2, c=150)
        self.block2 = IFBlock(10, scale=1, c=90)

    def forward(self, x):
        flow0 = self.block0(x)
        F1 = flow0
        F1_large = F.interpolate(F1, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_img0 = warp(x[:, :3], F1_large[:, :2])
        warped_img1 = warp(x[:, 3:], F1_large[:, 2:4])
        flow1 = self.block1(torch.cat((warped_img0, warped_img1, F1_large), 1))
        F2 = (flow0 + flow1)
        F2_large = F.interpolate(F2, scale_factor=2.0, mode="bilinear", align_corners=False) * 2.0
        warped_img0 = warp(x[:, :3], F2_large[:, :2])
        warped_img1 = warp(x[:, 3:], F2_large[:, 2:4])
        flow2 = self.block2(torch.cat((warped_img0, warped_img1, F2_large), 1))
        F3 = (flow0 + flow1 + flow2)

        return F3, [F1, F2, F3]


class FlowRefineNetA(nn.Module):
    def __init__(self, context_dim, c=16, r=1, n_iters=4):
        super(FlowRefineNetA, self).__init__()
        corr_dim = c
        flow_dim = c
        motion_dim = c
        hidden_dim = c

        self.n_iters = n_iters
        self.r = r
        self.n_pts = (r * 2 + 1) ** 2

        self.occl_convs = nn.Sequential(nn.Conv2d(2 * context_dim, hidden_dim, 1, 1, 0),
                                        nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
                                        nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, 1, 1, 1, 0),
                                        nn.Sigmoid())

        self.corr_convs = nn.Sequential(nn.Conv2d(self.n_pts, hidden_dim, 1, 1, 0),
                                        nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, corr_dim, 1, 1, 0),
                                        nn.PReLU(corr_dim))

        self.flow_convs = nn.Sequential(nn.Conv2d(2, hidden_dim, 3, 1, 1),
                                        nn.PReLU(hidden_dim),
                                        nn.Conv2d(hidden_dim, flow_dim, 3, 1, 1),
                                        nn.PReLU(flow_dim))

        self.motion_convs = nn.Sequential(nn.Conv2d(corr_dim + flow_dim, motion_dim, 3, 1, 1),
                                          nn.PReLU(motion_dim))

        self.gru = nn.Sequential(nn.Conv2d(motion_dim + context_dim * 2 + 2, hidden_dim, 3, 1, 1),
                                 nn.PReLU(hidden_dim),
                                 nn.Conv2d(hidden_dim, flow_dim, 3, 1, 1),
                                 nn.PReLU(flow_dim), )

        self.flow_head = nn.Sequential(nn.Conv2d(flow_dim, hidden_dim, 3, 1, 1),
                                       nn.PReLU(hidden_dim),
                                       nn.Conv2d(hidden_dim, 2, 3, 1, 1))

    def L2normalize(self, x, dim=1):
        eps = 1e-12
        norm = x ** 2
        norm = norm.sum(dim=dim, keepdim=True) + eps
        norm = norm ** (0.5)
        return (x/norm)

    def forward_once(self, x0, x1, flow0, flow1):
        B, C, H, W = x0.size()

        x0_unfold = F.unfold(x0, kernel_size=(self.r * 2 + 1), padding=1).view(B, C * self.n_pts, H,
                                                                               W)  # (B, C*n_pts, H, W)
        x1_unfold = F.unfold(x1, kernel_size=(self.r * 2 + 1), padding=1).view(B, C * self.n_pts, H,
                                                                               W)  # (B, C*n_pts, H, W)
        contents0 = warp(x0_unfold, flow0)
        contents1 = warp(x1_unfold, flow1)

        contents0 = contents0.view(B, C, self.n_pts, H, W)
        contents1 = contents1.view(B, C, self.n_pts, H, W)

        fea0 = contents0[:, :, self.n_pts // 2, :, :]
        fea1 = contents1[:, :, self.n_pts // 2, :, :]

        # get context feature
        occl = self.occl_convs(torch.cat([fea0, fea1], dim=1))
        fea = fea0 * occl + fea1 * (1 - occl)

        # get correlation features
        fea_view = fea.permute(0, 2, 3, 1).contiguous().view(B * H * W, 1, C)
        contents0 = contents0.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, self.n_pts, C)
        contents1 = contents1.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, self.n_pts, C)

        fea_view = self.L2normalize(fea_view, dim=-1)
        contents0 = self.L2normalize(contents0, dim=-1)
        contents1 = self.L2normalize(contents1, dim=-1)
        corr0 = torch.einsum('bic,bjc->bij', fea_view, contents0)  # (B*H*W, 1, n_pts)
        corr1 = torch.einsum('bic,bjc->bij', fea_view, contents1)
        # corr0 = corr0 / torch.sqrt(torch.tensor(C).float())
        # corr1 = corr1 / torch.sqrt(torch.tensor(C).float())
        corr0 = corr0.view(B, H, W, self.n_pts).permute(0, 3, 1, 2).contiguous()  # (B, n_pts, H, W)
        corr1 = corr1.view(B, H, W, self.n_pts).permute(0, 3, 1, 2).contiguous()
        corr0 = self.corr_convs(corr0)  # (B, corr_dim, H, W)
        corr1 = self.corr_convs(corr1)

        # get flow features
        flow0_fea = self.flow_convs(flow0)
        flow1_fea = self.flow_convs(flow1)

        # merge correlation and flow features, get motion features
        motion0 = self.motion_convs(torch.cat([corr0, flow0_fea], dim=1))
        motion1 = self.motion_convs(torch.cat([corr1, flow1_fea], dim=1))

        # update flows
        inp0 = torch.cat([fea, fea0, motion0, flow0], dim=1)
        delta_flow0 = self.flow_head(self.gru(inp0))
        flow0 = flow0 + delta_flow0
        inp1 = torch.cat([fea, fea1, motion1, flow1], dim=1)
        delta_flow1 = self.flow_head(self.gru(inp1))
        flow1 = flow1 + delta_flow1

        return flow0, flow1

    def forward(self, x0, x1, flow0, flow1):
        for i in range(self.n_iters):
            flow0, flow1 = self.forward_once(x0, x1, flow0, flow1)

        return torch.cat([flow0, flow1], dim=1)


class FlowRefineNet_Multis(nn.Module):
    def __init__(self, c=24, n_iters=1):
        super(FlowRefineNet_Multis, self).__init__()

        self.conv1 = Conv2(3, c, 1)
        self.conv2 = Conv2(c, 2 * c)
        self.conv3 = Conv2(2 * c, 4 * c)
        self.conv4 = Conv2(4 * c, 8 * c)

        self.rf_block1 = FlowRefineNetA(context_dim=c, c=c, r=1, n_iters=n_iters)
        self.rf_block2 = FlowRefineNetA(context_dim=2 * c, c=2 * c, r=1, n_iters=n_iters)
        self.rf_block3 = FlowRefineNetA(context_dim=4 * c, c=4 * c, r=1, n_iters=n_iters)
        self.rf_block4 = FlowRefineNetA(context_dim=8 * c, c=8 * c, r=1, n_iters=n_iters)

    def get_context(self, x0, x1, flow):
        bs = x0.size(0)

        inp = torch.cat([x0, x1], dim=0)
        s_1 = self.conv1(inp)  # 1
        s_2 = self.conv2(s_1)  # 1/2
        s_3 = self.conv3(s_2)  # 1/4
        s_4 = self.conv4(s_3)  # 1/8

        # warp features by the updated flow
        c0 = [s_1[:bs], s_2[:bs], s_3[:bs], s_4[:bs]]
        c1 = [s_1[bs:], s_2[bs:], s_3[bs:], s_4[bs:]]
        out0 = self.warp_fea(c0, flow[:, :2])
        out1 = self.warp_fea(c1, flow[:, 2:4])

        return flow, out0, out1

    def forward(self, x0, x1, flow):
        bs = x0.size(0)

        inp = torch.cat([x0, x1], dim=0)
        s_1 = self.conv1(inp)  # 1
        s_2 = self.conv2(s_1)  # 1/2
        s_3 = self.conv3(s_2)  # 1/4
        s_4 = self.conv4(s_3)  # 1/8

        # update flow from small scale
        flow = F.interpolate(flow, scale_factor=0.25, mode="bilinear", align_corners=False) * 0.25  # 1/8
        flow = self.rf_block4(s_4[:bs], s_4[bs:], flow[:, :2], flow[:, 2:4])  # 1/8
        flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.
        flow = self.rf_block3(s_3[:bs], s_3[bs:], flow[:, :2], flow[:, 2:4])  # 1/4
        flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.
        flow = self.rf_block2(s_2[:bs], s_2[bs:], flow[:, :2], flow[:, 2:4])  # 1/2
        flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.
        flow = self.rf_block1(s_1[:bs], s_1[bs:], flow[:, :2], flow[:, 2:4])  # 1

        # warp features by the updated flow
        c0 = [s_1[:bs], s_2[:bs], s_3[:bs], s_4[:bs]]
        c1 = [s_1[bs:], s_2[bs:], s_3[bs:], s_4[bs:]]
        out0 = self.warp_fea(c0, flow[:, :2])
        out1 = self.warp_fea(c1, flow[:, 2:4])

        return flow, out0, out1

    def warp_fea(self, feas, flow):
        outs = []
        for i, fea in enumerate(feas):
            out = warp(fea, flow)
            outs.append(out)
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        return outs



class VFIformer(nn.Module):
    def __init__(self, args):
        super(VFIformer, self).__init__()
        self.phase = args.phase
        self.device = args.device
        c = 24
        height = args.crop_size
        width = args.crop_size
        window_size = 8
        embed_dim = 160

        self.flownet = IFNet()
        self.refinenet = FlowRefineNet_Multis(c=c, n_iters=1)
        self.fuse_block = nn.Sequential(nn.Conv2d(12, 2*c, 3, 1, 1),
                                         nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                         nn.Conv2d(2*c, 2*c, 3, 1, 1),
                                         nn.LeakyReLU(negative_slope=0.2, inplace=True),)

        self.transformer = TFModel(img_size=(height, width), in_chans=2*c, out_chans=4, fuse_c=c,
                                          window_size=window_size, img_range=1.,
                                          depths=[[3, 3], [3, 3], [3, 3], [1, 1]],
                                          embed_dim=embed_dim, num_heads=[[2, 2], [2, 2], [2, 2], [2, 2]], mlp_ratio=2,
                                          resi_connection='1conv',
                                          use_crossattn=[[[False, False, False, False], [True, True, True, True]], \
                                                      [[False, False, False, False], [True, True, True, True]], \
                                                      [[False, False, False, False], [True, True, True, True]], \
                                                      [[False, False, False, False], [False, False, False, False]]])


        self.apply(self._init_weights)

        if args.resume_flownet:
            self.load_networks('flownet', args.resume_flownet)
            print('------ flownet loaded --------')

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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_flow(self, img0, img1):
        imgs = torch.cat((img0, img1), 1)
        flow, flow_list = self.flownet(imgs)
        flow, c0, c1 = self.refinenet(img0, img1, flow)

        return flow

    def forward(self, img0, img1, flow_pre=None):
        B, _, H, W = img0.size()
        imgs = torch.cat((img0, img1), 1)

        if flow_pre is not None:
            flow = flow_pre
            _, c0, c1 = self.refinenet.get_context(img0, img1, flow)
        else:
            flow, flow_list = self.flownet(imgs)
            flow, c0, c1 = self.refinenet(img0, img1, flow)


        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:])
        
        x = self.fuse_block(torch.cat([img0, img1, warped_img0, warped_img1], dim=1))

        refine_output = self.transformer(x, c0, c1)
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = warped_img0 * mask + warped_img1 * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)

        if self.phase == 'train':
            return pred, flow_list
        else:
            return pred, flow




#-------------------------------------
# light-weight version
#-------------------------------------

class FlowRefineNet_Multis_Simple(nn.Module):
    def __init__(self, c=24, n_iters=1):
        super(FlowRefineNet_Multis_Simple, self).__init__()

        self.conv1 = Conv2(3, c, 1)
        self.conv2 = Conv2(c, 2 * c)
        self.conv3 = Conv2(2 * c, 4 * c)
        self.conv4 = Conv2(4 * c, 8 * c)

    def forward(self, x0, x1, flow):
        bs = x0.size(0)

        inp = torch.cat([x0, x1], dim=0)
        s_1 = self.conv1(inp)  # 1
        s_2 = self.conv2(s_1)  # 1/2
        s_3 = self.conv3(s_2)  # 1/4
        s_4 = self.conv4(s_3)  # 1/8

        flow = F.interpolate(flow, scale_factor=2., mode="bilinear", align_corners=False) * 2.

        # warp features by the updated flow
        c0 = [s_1[:bs], s_2[:bs], s_3[:bs], s_4[:bs]]
        c1 = [s_1[bs:], s_2[bs:], s_3[bs:], s_4[bs:]]
        out0 = self.warp_fea(c0, flow[:, :2])
        out1 = self.warp_fea(c1, flow[:, 2:4])

        return flow, out0, out1

    def warp_fea(self, feas, flow):
        outs = []
        for i, fea in enumerate(feas):
            out = warp(fea, flow)
            outs.append(out)
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False) * 0.5
        return outs



class VFIformerSmall(nn.Module):
    def __init__(self, args):
        super(VFIformerSmall, self).__init__()
        self.phase = args.phase
        self.device = args.device
        c = 24
        height = args.crop_size
        width = args.crop_size
        window_size = 4
        embed_dim = 136

        self.flownet = IFNet()
        self.refinenet = FlowRefineNet_Multis_Simple(c=c, n_iters=1)
        self.fuse_block = nn.Sequential(nn.Conv2d(12, 2*c, 3, 1, 1),
                                         nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                         nn.Conv2d(2*c, 2*c, 3, 1, 1),
                                         nn.LeakyReLU(negative_slope=0.2, inplace=True),)

        self.transformer = TFModel(img_size=(height, width), in_chans=2*c, out_chans=4, fuse_c=c,
                                          window_size=window_size, img_range=1.,
                                          depths=[[3, 3], [3, 3], [3, 3], [1, 1]],
                                          embed_dim=embed_dim, num_heads=[[2, 2], [2, 2], [2, 2], [2, 2]], mlp_ratio=2,
                                          resi_connection='1conv',
                                          use_crossattn=[[[False, False, False, False], [True, True, True, True]], \
                                                      [[False, False, False, False], [True, True, True, True]], \
                                                      [[False, False, False, False], [True, True, True, True]], \
                                                      [[False, False, False, False], [False, False, False, False]]])


        self.apply(self._init_weights)

        if args.resume_flownet:
            self.load_networks('flownet', args.resume_flownet)
            print('------ flownet loaded --------')

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

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_flow(self, img0, img1):
        imgs = torch.cat((img0, img1), 1)
        flow, flow_list = self.flownet(imgs)
        flow, c0, c1 = self.refinenet(img0, img1, flow)

        return flow

    def forward(self, img0, img1, flow_pre=None):
        B, _, H, W = img0.size()
        imgs = torch.cat((img0, img1), 1)

        if flow_pre is not None:
            flow = flow_pre
            _, c0, c1 = self.refinenet(img0, img1, flow)

        else:
            flow, flow_list = self.flownet(imgs)
            flow, c0, c1 = self.refinenet(img0, img1, flow)


        warped_img0 = warp(img0, flow[:, :2])
        warped_img1 = warp(img1, flow[:, 2:])
        
        x = self.fuse_block(torch.cat([img0, img1, warped_img0, warped_img1], dim=1))

        refine_output = self.transformer(x, c0, c1)
        res = torch.sigmoid(refine_output[:, :3]) * 2 - 1
        mask = torch.sigmoid(refine_output[:, 3:4])
        merged_img = warped_img0 * mask + warped_img1 * (1 - mask)
        pred = merged_img + res
        pred = torch.clamp(pred, 0, 1)

        if self.phase == 'train':
            return pred, flow_list
        else:
            return pred, flow






if __name__ == "__main__":
    # try:
    #     from models.archs.dcn.deform_conv import ModulatedDeformConvPack as DCN
    # except ImportError:
    #     raise ImportError('Failed to import DCNv2 module.')

    import argparse
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--phase', default='train', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--crop_size', default=192, type=int)
    args = parser.parse_args()

    device = 'cuda'

    net = Swin_Fuse_CrossScaleV2_MaskV5_Normal_WoRefine_ConvBaseline(args).to(device)
    print('----- generator parameters: %f -----' % (sum(param.numel() for param in net.parameters()) / (10**6)))
    
    w = 192
    img0 = torch.randn((2, 3, w, w)).to(device)
    img1 = torch.randn((2, 3, w, w)).to(device)
    out = net(img0, img1)
    print(out[0].size())
