import os
import math
import cv2
import argparse
from torch.nn.parallel import DistributedDataParallel
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
from skimage.color import rgb2yuv, yuv2rgb
from utils.util import setup_logger, print_args
from utils.pytorch_msssim import ssim_matlab
from models.modules import define_G
from tqdm import tqdm

# with one split the secondary tqdm is not needed, with verbose both not needed
# encode the frame set being worked on in the temporary files for use in inspection

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
    parser = argparse.ArgumentParser(description='infinite division of video frames')
    parser.add_argument('--model', default='./pretrained_models/pretrained_VFIformer/net_220.pth', type=str)
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--save_folder', default='./output', type=str)
    parser.add_argument('--base_path', default='./images', type=str, help="path to png files")
    parser.add_argument('--base_name', default='image', type=str, help="filename before 0-filled index number")
    parser.add_argument('--img_first', default=0, type=int, help="first image index")
    parser.add_argument('--img_last', default=2, type=int, help="last image index")
    parser.add_argument('--num_width', default=1, type=int, help="index width for zero filling")
    parser.add_argument('--num_splits', default=2, type=int, help="how many doublings of the pool of frames")
    parser.add_argument("--verbose", dest="verbose", default=False, action="store_true", help="Show extra details")

    ## setup training environment
    args = parser.parse_args()

    init_log(args.verbose)

    ## setup training device
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)
    if len(args.gpu_ids) > 0:
        torch.cuda.set_device(args.gpu_ids[0])

    cudnn.benchmark = True

    ## save paths
    save_path = args.save_folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # defaults instead of unneeded arguments
    args.crop_size = 192
    args.dist = False
    args.rank = -1
    args.phase = "test"
    args.resume_flownet = ""
    args.net_name = "VFIformer"

    ## load model
    device = torch.device('cuda' if len(args.gpu_ids) != 0 else 'cpu')
    args.device = device
    net = define_G(args)
    net = load_networks(net, args.model)
    net.eval()

    basepath = args.base_path
    basefile = args.base_name
    start = args.img_first
    end = args.img_last
    num_width = args.num_width
    working_prefix = save_path + "\\" + basefile
    for n in tqdm(range(start, end), desc="Total", position=1):
        continued = n > start
        split_frames(net, args.num_splits, basepath, basefile, n, n+1, num_width, working_prefix, save_path, continued)

def split_frames(net, num_splits, basepath, basefile, start, end, num_width, working_prefix, save_path, continued):
    init_record()
    reset_split_count(num_splits)

    # 2 to the power of the number of doublings, origin zero
    max_steps = 2 ** num_splits - 1
    init_progress(max_steps, "Frame #" + str(start + 1))

    first_file = basepath + "\\" + basefile + str(start).zfill(num_width) + ".png"
    last_file = basepath + "\\" + basefile + str(end).zfill(num_width) + ".png"
    img0 = cv2.imread(first_file)
    img1 = cv2.imread(last_file)

    # create 0.0 and 1.0 versions of the outer real frames
    first_index = 0.0
    last_index = 1.0
    first_file = working_filepath(working_prefix, first_index)
    last_file = working_filepath(working_prefix, last_index)

    cv2.imwrite(first_file, img0)
    record_frame(first_index)
    log("main() saved " + first_file)

    cv2.imwrite(last_file, img1)
    record_frame(last_index)
    log("main() saved " + last_file)

    recursive_split_frames(net, first_index, last_index, working_prefix)
    integerize_filenames(working_prefix, save_path, basefile, start, end, continued, num_width)
    close_progress()

def recursive_split_frames(net, first_index, last_index, filepath_prefix):
    if enter_split():
        mid_index = first_index + (last_index - first_index) / 2.0

        # first_filepath = filepath_prefix + str(first_index) + ".png"
        # last_filepath = filepath_prefix + str(last_index) + ".png"
        # mid_filepath = filepath_prefix + str(mid_index) + ".png"

        first_filepath = working_filepath(filepath_prefix, first_index)
        last_filepath = working_filepath(filepath_prefix, last_index)
        mid_filepath = working_filepath(filepath_prefix, mid_index)

        create_mid_frame(net, first_filepath, last_filepath, mid_filepath)
        record_frame(mid_index)
        step_progress()

        # deal with two new split regions
        recursive_split_frames(net, first_index, mid_index, filepath_prefix)
        recursive_split_frames(net, mid_index, last_index, filepath_prefix)
        exit_split()

def create_mid_frame(net, first_filepath, last_filepath, mid_filepath):
    img0 = cv2.imread(first_filepath)
    img1 = cv2.imread(last_filepath)

    divisor = 64
    h, w, _ = img0.shape
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
    torchvision.utils.save_image(imt, mid_filepath)
    log("create_mid_frame() saved " + mid_filepath)

def integerize_filenames(working_filepath_prefix, save_path, base_name, start, end, continued, num_width):
    new_prefix = save_path + "//" + base_name + "[" + str(start).zfill(num_width) + "-" + str(end).zfill(num_width) + "]"

    frames = sorted_frames()
    this_round_num_width = len(str(len(frames)))

    index = 0
    for f in sorted_frames():
        # orig_filename = working_filepath_prefix + str(f) + ".png"
        orig_filename = working_filepath(working_filepath_prefix, f)

        if continued and index == 0:
            # if a continuation from a previous set of frames, delete the first frame
            # since it's duplicate of the previous round last frame
            os.remove(orig_filename)
            log("integerize_filenames() removed uneeded " + orig_filename)
        else:
            new_filename = new_prefix + str(index).zfill(this_round_num_width) + ".png"
            os.replace(orig_filename, new_filename)
            log("integerize_filenames() renamed " + orig_filename + " to " + new_filename)

        index += 1

global split_count
def reset_split_count(num_splits):
    global split_count
    split_count = num_splits

def enter_split():
    global split_count
    if split_count < 1:
        return False
    split_count -= 1
    return True

def exit_split():
    global split_count
    split_count += 1

global frame_record
def init_record():
    global frame_record
    frame_record = []

def record_frame(index):
    global frame_record
    frame_record.append(index)

def sorted_frames():
    global frame_record
    return sorted(frame_record)

global verbose
def init_log(verbose_enabled):
    global verbose
    verbose = verbose_enabled

def log(message):
    if verbose:
        print(message)

global split_progress
def init_progress(max, description):
    global split_progress
    split_progress = tqdm(range(max), desc=description)

def step_progress():
    global split_progress
    split_progress.update()
    split_progress.refresh()

def close_progress():
    global split_progress
    split_progress.close()

def working_filepath(filepath_prefix, index):
    return filepath_prefix + f"{index:1.24f}.png"


if __name__ == '__main__':
    main()


