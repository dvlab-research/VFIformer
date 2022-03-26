import os
import numpy as np
import glob
import cv2
import sys
import random
import math
import torch
from torch.utils.data import Dataset

# sys.path.append('..')
# from utils import util


cv2.setNumThreads(1)
class VimeoDataset(Dataset):
    def __init__(self, args):
        self.data_root = args.data_root
        self.phase = args.phase
        self.crop_size = args.crop_size

        self.load_data()

    def __len__(self):
        return len(self.meta_data)

    def load_data(self):
        self.meta_data = []
        self.flow_data = []
        self.folder_data = []
        if self.phase == 'train':
            data_list = open(os.path.join(self.data_root, 'tri_trainlist.txt'), 'r')
        else:
            data_list = open(os.path.join(self.data_root, 'tri_testlist.txt'), 'r')

        for item in data_list:
            name = str(item).strip()
            if(len(name) <= 1):
                continue
            pair = sorted(glob.glob(os.path.join(self.data_root, 'sequences', name, '*.png')))
            flow = sorted(glob.glob(os.path.join(self.data_root.replace('vimeo_triplet', ''), 'flows', name, '*.npy')))
            self.meta_data.append(pair)
            self.flow_data.append(flow)
            self.folder_data.append(name)

        self.nr_sample = len(self.meta_data)        

    def aug(self, img0, gt, img1, flow_gt, h, w):
        ih, iw, _ = img0.shape
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        img0 = img0[x:x+h, y:y+w, :]
        img1 = img1[x:x+h, y:y+w, :]
        gt = gt[x:x+h, y:y+w, :]
        flow_gt = flow_gt[x:x+h, y:y+w, :]
        return img0, gt, img1, flow_gt

    def getimg(self, index):
        data = self.meta_data[index]
        flow = self.flow_data[index]
        folder = self.folder_data[index]

        img0 = cv2.imread(data[0])
        gt = cv2.imread(data[1])
        img1 = cv2.imread(data[2])

        flow0 = np.load(flow[0])
        flow1 = np.load(flow[1])
        flow_gt = np.concatenate([flow0, flow1], axis=0).transpose(1, 2, 0)

        return img0, gt, img1, flow_gt, folder
            
    def __getitem__(self, index):        
        img0, gt, img1, flow_gt, folder = self.getimg(index)
        if self.phase == 'train':
            img0, gt, img1, flow_gt = self.aug(img0, gt, img1, flow_gt, self.crop_size, self.crop_size)

            # # attention: can only be used without flow loss !!!
            # if random.uniform(0, 1) < 0.5:  # rotate
            #     img0 = np.ascontiguousarray(np.rot90(img0, k=1, axes=(0, 1)))
            #     img1 = np.ascontiguousarray(np.rot90(img1, k=1, axes=(0, 1)))
            #     gt = np.ascontiguousarray(np.rot90(gt, k=1, axes=(0, 1)))

            # if random.uniform(0, 1) < 0.5:  # color aug
            #     img0 = img0[:, :, ::-1]
            #     img1 = img1[:, :, ::-1]
            #     gt = gt[:, :, ::-1]
            if random.uniform(0, 1) < 0.5:  # vertical flip
                img0 = img0[::-1]
                img1 = img1[::-1]
                gt = gt[::-1]
                flow_gt = flow_gt[::-1]
                flow_gt = np.concatenate((flow_gt[:, :, 0:1], -flow_gt[:, :, 1:2], flow_gt[:, :, 2:3], -flow_gt[:, :, 3:4]), 2)
            if random.uniform(0, 1) < 0.5:  # horizontal flip
                img0 = img0[:, ::-1]
                img1 = img1[:, ::-1]
                gt = gt[:, ::-1]
                flow_gt = flow_gt[:, ::-1]
                flow_gt = np.concatenate((-flow_gt[:, :, 0:1], flow_gt[:, :, 1:2], -flow_gt[:, :, 2:3], flow_gt[:, :, 3:4]), 2)
            if random.uniform(0, 1) < 0.5:  # reverse time
                tmp = img1
                img1 = img0
                img0 = tmp
                flow_gt = np.concatenate((flow_gt[:, :, 2:4], flow_gt[:, :, 0:2]), 2)
        else:
            h, w, _ = img0.shape
            # flow_gt = np.zeros((h, w, 4))

        if self.phase == 'train':
            flow_gt = torch.from_numpy(flow_gt).float().permute(2, 0, 1)
            img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1)
            gt = torch.from_numpy(gt.astype('float32') / 255.).float().permute(2, 0, 1)
            img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1)

            sample = {'img0': img0,
                      'img1': img1,
                      'gt': gt,
                      'flow_gt': flow_gt,
                      'folder': folder}
        else:
            # pad HR to be mutiple of 64
            h, w, c = gt.shape
            if h % 64 != 0 or w % 64 != 0:
                h_new = math.ceil(h / 64) * 64
                w_new = math.ceil(w / 64) * 64
                pad_t = (h_new - h) // 2
                pad_d = (h_new - h) // 2 + (h_new - h) % 2
                pad_l = (w_new - w) // 2
                pad_r = (w_new - w) // 2 + (w_new - w) % 2
                img0 = cv2.copyMakeBorder(img0.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)  # cv2.BORDER_REFLECT
                img1 = cv2.copyMakeBorder(img1.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
            else:
                pad_t, pad_d, pad_l, pad_r = 0, 0, 0, 0
            pad_nums = [pad_t, pad_d, pad_l, pad_r]

            flow_gt = torch.from_numpy(flow_gt).float().permute(2, 0, 1)
            img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1)
            gt = torch.from_numpy(gt).permute(2, 0, 1)
            # gt = torch.from_numpy(gt.astype('float32') / 255.).float().permute(2, 0, 1)
            img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1)

            sample = {'img0': img0,
                      'img1': img1,
                      'gt': gt,
                      'flow_gt': flow_gt,
                      'pad_nums': pad_nums,
                      'folder': folder}

        return sample


class MiddleburyDataset(Dataset):
    def __init__(self, args):
        self.data_root = args.data_root
        # self.data_root = '/home/liyinglu/newData/datasets/vfi/Middlebury/'
        self.load_data()

    def __len__(self):
        return len(self.data_list)

    def load_data(self):
        name = ['Beanbags', 'Dimetrodon', 'DogDance', 'Grove2', 'Grove3', 'Hydrangea', 'MiniCooper', 'RubberWhale',
                'Urban2', 'Urban3', 'Venus', 'Walking']
        data_list = []
        for i in name:
            img0 = os.path.join(self.data_root, 'other-data/{}/frame10.png'.format(i))
            img1 = os.path.join(self.data_root, 'other-data/{}/frame11.png'.format(i))
            gt = os.path.join(self.data_root, 'other-gt-interp/{}/frame10i11.png'.format(i))
            data_list.append([img0, img1, gt, i])

        self.data_list = data_list

    def __getitem__(self, index):
        img0 = cv2.imread(self.data_list[index][0])
        img1 = cv2.imread(self.data_list[index][1])
        gt = cv2.imread(self.data_list[index][2])

        # pad HR to be mutiple of 64
        h, w, c = gt.shape
        if h % 64 != 0 or w % 64 != 0:
            h_new = math.ceil(h / 64) * 64
            w_new = math.ceil(w / 64) * 64
            pad_t = (h_new - h) // 2
            pad_d = (h_new - h) // 2 + (h_new - h) % 2
            pad_l = (w_new - w) // 2
            pad_r = (w_new - w) // 2 + (w_new - w) % 2
            img0 = cv2.copyMakeBorder(img0.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)  # cv2.BORDER_REFLECT
            img1 = cv2.copyMakeBorder(img1.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        else:
            pad_t, pad_d, pad_l, pad_r = 0, 0, 0, 0
        pad_nums = [pad_t, pad_d, pad_l, pad_r]


        img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1)
        gt = torch.from_numpy(gt).permute(2, 0, 1)
        # gt = torch.from_numpy(gt.astype('float32') / 255.).float().permute(2, 0, 1)
        img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1)

        sample = {'img0': img0,
                  'img1': img1,
                  'gt': gt,
                  'pad_nums': pad_nums,
                  'folder': self.data_list[index][3]}

        return sample


class UFC101Dataset(Dataset):
    def __init__(self, args):
        self.data_root = args.data_root
        # self.data_root = '/home/liyinglu/newData/datasets/vfi/ucf101_interp_ours/'
        self.load_data()

    def __len__(self):
        return len(self.data_list)

    def load_data(self):
        dirs = os.listdir(self.data_root)
        data_list = []
        for d in dirs:
            img0 = os.path.join(self.data_root, d, 'frame_00.png')
            img1 = os.path.join(self.data_root, d, 'frame_02.png')
            gt = os.path.join(self.data_root, d, 'frame_01_gt.png')
            data_list.append([img0, img1, gt, d])

        self.data_list = data_list

    def __getitem__(self, index):
        img0 = cv2.imread(self.data_list[index][0])
        img1 = cv2.imread(self.data_list[index][1])
        gt = cv2.imread(self.data_list[index][2])

        # pad HR to be mutiple of 64
        h, w, c = gt.shape
        if h % 64 != 0 or w % 64 != 0:
            h_new = math.ceil(h / 64) * 64
            w_new = math.ceil(w / 64) * 64
            pad_t = (h_new - h) // 2
            pad_d = (h_new - h) // 2 + (h_new - h) % 2
            pad_l = (w_new - w) // 2
            pad_r = (w_new - w) // 2 + (w_new - w) % 2
            img0 = cv2.copyMakeBorder(img0.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)  # cv2.BORDER_REFLECT
            img1 = cv2.copyMakeBorder(img1.copy(), pad_t, pad_d, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        else:
            pad_t, pad_d, pad_l, pad_r = 0, 0, 0, 0
        pad_nums = [pad_t, pad_d, pad_l, pad_r]


        img0 = torch.from_numpy(img0.astype('float32') / 255.).float().permute(2, 0, 1)
        gt = torch.from_numpy(gt).permute(2, 0, 1)
        # gt = torch.from_numpy(gt.astype('float32') / 255.).float().permute(2, 0, 1)
        img1 = torch.from_numpy(img1.astype('float32') / 255.).float().permute(2, 0, 1)

        sample = {'img0': img0,
                  'img1': img1,
                  'gt': gt,
                  'pad_nums': pad_nums,
                  'folder': self.data_list[index][3]}

        return sample

