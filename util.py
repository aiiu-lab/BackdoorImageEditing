from ast import Slice
from dataclasses import dataclass
from math import ceil, floor, sqrt
import os
from datetime import datetime
from typing import Tuple, Union, Dict, List
import warnings
import math
import PIL

import psutil
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.dates import SA
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
from PIL import Image
from comet_ml import Experiment, ExistingExperiment

from models.StegaStamp import StegaStampEncoder, StegaStampDecoder


def normalize(x: Union[np.ndarray, torch.Tensor], vmin_in: float=None, vmax_in: float=None, vmin_out: float=0, vmax_out: float=1, eps: float=1e-5) -> Union[np.ndarray, torch.Tensor]:
    if vmax_out == None and vmin_out == None:
        return x

    if isinstance(x, np.ndarray):
        if vmin_in == None:
            min_x = np.min(x)
        else:
            min_x = vmin_in
        if vmax_in == None:
            max_x = np.max(x)
        else:
            max_x = vmax_in
    elif isinstance(x, torch.Tensor):
        if vmin_in == None:
            min_x = torch.min(x)
        else:
            min_x = vmin_in
        if vmax_in == None:
            max_x = torch.max(x)
        else:
            max_x = vmax_in
    else:
        raise TypeError("x must be a torch.Tensor or a np.ndarray")
    if vmax_out == None:
        vmax_out = max_x
    if vmin_out == None:
        vmin_out = min_x
    return ((x - min_x) / (max_x - min_x + eps)) * (vmax_out - vmin_out) + vmin_out


GREY_BG_RATIO = 0.3

def bg2gray(tensor, vmax=1, vmin=-1):
    thres = (vmax - vmin) * GREY_BG_RATIO + vmin
    tensor[tensor <= thres] = thres
    return tensor

def generate_bitstring_watermark(bs, bit_length):
    msg = torch.randint(0, 2, (bs, bit_length)).float()
    return msg

def load_stegastamp_encoder(args):
    state_dict = torch.load(args.encoder_path,map_location="cuda")
    fingerpint_size = state_dict["secret_dense.weight"].shape[-1]

    HideNet = StegaStampEncoder(
        args.resolution,
        3,
        fingerprint_size=fingerpint_size,
        return_residual=False,
    )

    HideNet.load_state_dict(state_dict)

    return HideNet, fingerpint_size

def load_stegastamp_decoder(args):
    
    state_dict = torch.load(args.decoder_path,map_location="cuda")
    fingerprint_size = state_dict["dense.2.weight"].shape[0]

    RevealNet = StegaStampDecoder(args.resolution, 3, fingerprint_size)
    RevealNet.load_state_dict(state_dict)
    
    return RevealNet

def tensor_to_pil(tensor):
    # from tensor [-1, 1] transfer to [0,255] and modify dim
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor + 1) / 2 * 255
    tensor = np.clip(tensor, 0, 255).astype(np.uint8)
    # (3, H, W) -> (H, W, 3)
    tensor = np.transpose(tensor, (1, 2, 0))
    return PIL.Image.fromarray(tensor)

def create_grid(image_list, ncols=10):
    """使用固定列數生成網格圖像"""
    if not image_list:
        return None
    # 假設所有圖片大小一致
    w, h = image_list[0].size
    n_images = len(image_list)
    nrows = math.ceil(n_images / ncols)
    grid_img = PIL.Image.new('RGB', (w * ncols, h * nrows))
    for idx, img in enumerate(image_list):
        col = idx % ncols
        row = idx // ncols
        grid_img.paste(img, (col * w, row * h))
    return grid_img

def concat_grids(grid1, grid2):
    if grid1 is None or grid2 is None:
        return None
    total_height = grid1.height + grid2.height
    max_width = max(grid1.width, grid2.width)
    new_img = PIL.Image.new('RGB', (max_width, total_height))
    new_img.paste(grid1, (0, 0))
    new_img.paste(grid2, (0,grid1.height))
    return new_img

def convert_to_np(image, resolution):
    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)
