#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to fine-tune Stable Diffusion for InstructPix2Pix."""
import time
import argparse
import logging
import math
import os, sys
import shutil
from contextlib import nullcontext
from pathlib import Path
import wandb
from datetime import datetime
import random
import re
import csv

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..")))

import accelerate
import datasets
import numpy as np
import PIL
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as func
import lpips  
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler, get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from util import load_stegastamp_encoder, load_rosteals_model, get_backdoor_target, convert_to_np, create_grid, generate_bitstring_watermark, tensor_to_pil
from models.StegaStamp import StegaStampEncoder, StegaStampDecoder
#from dataset import get_celeba_hq_dataset, ClelebAHQWatermarkedDataset, CelebADataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from clip_similarity import ClipSimilarity


logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
    "timbrooks/instructpix2pix-clip-filtered": ("original_image", "edit_prompt", "edited_image"),
}
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt", "watermark_image", "bd_edited_image"]


distortion_strength_paras = dict(
    rotation=(9, 45), # rotation angle (0,45)
    resizedcrop=(0.9, 0.5), # scale (1,0.5)
    erasing=(0.05, 0.25), # erasing scale (0,0.25)
    brightness=(1.2, 2),  # Brightness (1,2)
    contrast=(1.2, 2),    # Contrast (1,2)
    blurring=(4, 20),      # Gaussian Blur (0,20)
    noise=(0.02, 0.1),      # Gaussian Noise (0,0.1)
    compression=(90,10),  # JPEG (90,10)
    # dropout=0.3,     # Dropout (pixel dropout probability)
    # hue=0.2,         # Hue (normalized shift factor)
    # gif=16         # GIF effect (number of colors)
)

def image_distortion(images, distortion_type):
    if distortion_type == "clean":
        clean_images = []
        for i in range(images.size(0)):
            clean_images.append(tensor_to_pil(images[i]))
        return clean_images

    if distortion_type == "rotation":
        def distort(img):
            angle = random.uniform(*distortion_strength_paras["rotation"])
            return func.rotate(img, angle)
        
    elif distortion_type == "resizedcrop":
        def distort(img):
            scale = random.uniform(*distortion_strength_paras["resizedcrop"])
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img, scale=(scale, scale), ratio=(1, 1)
            )
            return  func.resized_crop(img, i, j, h, w, img.size)

    elif distortion_type == "erasing":
        def distort(img):
            scale = random.uniform(*distortion_strength_paras["erasing"])
            image_tensor = transforms.ToTensor()(img)
            i, j, h, w, v = transforms.RandomErasing.get_params(
                image_tensor, scale=(scale, scale), ratio=(1, 1), value=[0]
            )
            distorted_image = func.erase(image_tensor, i, j, h, w, v)
            return transforms.ToPILImage()(distorted_image)
    elif distortion_type == "brightness":
        def distort(img):
            factor = random.uniform(*distortion_strength_paras["brightness"])
            enhancer = ImageEnhance.Brightness(img)
            return enhancer.enhance(factor)
    elif distortion_type == "contrast":
        def distort(img):
            factor = random.uniform(*distortion_strength_paras["contrast"])
            enhancer = ImageEnhance.Contrast(img)
            return enhancer.enhance(factor)
    elif distortion_type == "blurring":
        def distort(img):
            kernel = random.uniform(*distortion_strength_paras["blurring"])
            return img.filter(ImageFilter.GaussianBlur(kernel))
    elif distortion_type == "noise":
        def distort(img):
            std = random.uniform(*distortion_strength_paras["noise"])
            image_tensor = transforms.ToTensor()(img)
            noise = torch.randn(image_tensor.size()) * std
            noisy_tensor = (image_tensor + noise).clamp(0, 1)
            return transforms.ToPILImage()(noisy_tensor)
    elif distortion_type == "compression":
        def distort(img):
            quality = random.randint(*distortion_strength_paras["compression"])
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=quality)
            buffered.seek(0)
            return Image.open(buffered)
    # elif distortion_type == "crop":
    #     # crop a portion (factor: 1 - crop_ratio) and then resize back to original size
    #     crop_ratio = distortion_strength_paras["crop"]
    #     def distort(img):
    #         w, h = img.size
    #         new_w = int(w * (1 - crop_ratio))
    #         new_h = int(h * (1 - crop_ratio))
    #         left = random.randint(0, w - new_w)
    #         top = random.randint(0, h - new_h)
    #         cropped = img.crop((left, top, left + new_w, top + new_h))
    #         return cropped.resize((w, h), resample=Image.BILINEAR)
    # elif distortion_type == "dropout":
    #     dropout_prob = distortion_strength_paras.get("dropout", 0.3)
    #     def distort(img):
    #         img_arr = np.array(img)
    #         # 若彩圖，shape 為 (H, W, C)
    #         mask = np.random.rand(img_arr.shape[0], img_arr.shape[1]) < dropout_prob
    #         # 對每個 channel 採用相同的 dropout mask
    #         img_arr[mask] = 0
    #         return Image.fromarray(img_arr)
    # elif distortion_type == "hue":
    #     hue_shift = distortion_strength_paras.get("hue", 0.2)
    #     # 將 hue_shift 轉換成 [0,255] 範圍的位移值
    #     shift_value = int(hue_shift * 255)
    #     def distort(img):
    #         hsv = np.array(img.convert("HSV"), dtype=np.uint8)
    #         hsv[..., 0] = (hsv[..., 0] + shift_value) % 256
    #         return Image.fromarray(hsv, mode="HSV").convert("RGB")
    # elif distortion_type == "saturation":
    #     saturation_factor = distortion_strength_paras.get("saturation", 15.0)
    #     def distort(img):
    #         enhancer = ImageEnhance.Color(img)
    #         return enhancer.enhance(saturation_factor)
    # elif distortion_type == "resize":
    #     resize_factor = distortion_strength_paras.get("resize", 0.7)
    #     def distort(img):
    #         w, h = img.size
    #         new_size = (int(w * resize_factor), int(h * resize_factor))
    #         resized = img.resize(new_size, resample=Image.BILINEAR)
    #         return resized.resize((w, h), resample=Image.BICUBIC)
    # elif distortion_type == "gif":
    #     # reduce the number of colors to create a GIF-like effect
    #     color_count = distortion_strength_paras.get("gif", 16)
    #     def distort(img):
    #         return img.convert("P", palette=Image.ADAPTIVE, colors=color_count).convert("RGB")
    else:
        raise ValueError(f"Unknown distortion type: {distortion_type}")

    distorted_images = []
    for i in range(images.size(0)):
        pil_img = tensor_to_pil(images[i])
        distorted_pil = distort(pil_img)
        distorted_images.append(distorted_pil)
    return distorted_images

def log_validation_set(args, pipeline, accelerator, eval_dataset, generator, epoch, output_dir, gt_msgs, visual_ncols=10):
    logger.info("Running validation on validation set samples (batch inference)")
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    max_log_num = args.max_log_num
    global_idx = 0  
    group_size = args.eval_samples // args.backdoor_target_num

    def collate_fn(examples):
        non_watermark_for_eval = torch.stack([example["non_watermark_for_eval"] for example in examples])
        non_watermark_for_eval = non_watermark_for_eval.to(memory_format=torch.contiguous_format).float()

        edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
        edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()

        raw_prompts = [example["raw_prompt"] for example in examples]
        edited_image_captions = [example["edited_image_caption"] for example in examples]
        original_image_captions = [example["original_image_caption"] for example in examples]
        
        return {
            "non_watermark_for_eval": non_watermark_for_eval,
            "raw_prompt": raw_prompts,
            "edited_pixel_values": edited_pixel_values,
            "edited_image_caption": edited_image_captions,
            "original_image_caption": original_image_captions
        }

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=10,
        num_workers=args.dataloader_num_workers,
    )

    # 用於可視化的圖片，只保留前 visualization_limit 張
    vis_original = []
    vis_wm = []
    vis_gtedited = []
    vis_edited = []
    vis_bd_edited = []

    visualization_limit = visual_ncols * visual_ncols
    
    original_image_list = []
    gtedited_image_list = []
    edited_image_list = []
    bd_edited_image_list = []
    # bit_acc_wm_list = []

    wandb_table = wandb.Table(columns=WANDB_TABLE_COL_NAMES)

    input_captions_all = []
    output_captions_all = []
    start_time = time.time()
    for batch in eval_dataloader:
        # 假設 batch["non_watermark_for_eval"] 為 (B, C, H, W) 且 batch["raw_prompt"] 為 list
        bs = batch["non_watermark_for_eval"].shape[0]
        # 將 tensor 轉成 PIL 圖片
        # original_imgs = [tensor_to_pil(img) for img in batch["non_watermark_for_eval"]]
        original_imgs = image_distortion(batch["non_watermark_for_eval"], args.distortion_type)
        prompts = batch["raw_prompt"] 
        output_captions_all.extend(batch["edited_image_caption"])
        input_captions_all.extend(batch["original_image_caption"])

        # 使用 pipeline 批次推論：編輯圖片
        with torch.autocast(accelerator.device.type):
            edited_imgs = pipeline(
                prompts,
                image=original_imgs,
                num_inference_steps=50,
                image_guidance_scale=1.5,
                guidance_scale=7,
                generator=generator
            ).images  # 應返回 list


        
        gtedited_imgs = [tensor_to_pil(img) for img in batch["edited_pixel_values"]]

        
        # 編碼 watermark：直接處理整個 batch
        selected_msgs_list = []
        selected_gt_list = []
        for j in range(bs):
            target_index = (global_idx + j) // group_size
            target_index = min(target_index, args.backdoor_target_num - 1)
            selected_msgs_list.append(msgs[target_index])
            selected_gt_list.append(gt_msgs[target_index])
            
        selected_msgs = torch.stack(selected_msgs_list).to(accelerator.device)
        wm_tensor = watermark_encoder(selected_msgs, batch["non_watermark_for_eval"].to(accelerator.device))
        wm_tensor = wm_tensor * 2 - 1
        # test robustness against distortion
        wm_imgs = image_distortion(wm_tensor, args.distortion_type)
        #wm_imgs = [tensor_to_pil(img) for img in wm_tensor]

        # Backdoor editing 批次推論
        with torch.autocast(accelerator.device.type):
            bd_edited_imgs = pipeline(
                prompts,
                image=wm_imgs,
                num_inference_steps=50,
                image_guidance_scale=1.5,
                guidance_scale=7,
                generator=generator
            ).images

        # 將 batch 中的結果拆解後累積
        original_image_list.extend(original_imgs)
        edited_image_list.extend(edited_imgs)
        bd_edited_image_list.extend(bd_edited_imgs)
        gtedited_image_list.extend(gtedited_imgs)

        # 如果全域索引還在 max_log_num 內，加入 wandb log
        for j in range(bs):
            if global_idx < visualization_limit:
                vis_original.append(original_imgs[j])
                vis_wm.append(wm_imgs[j])
                vis_gtedited.append(gtedited_imgs[j])
                vis_edited.append(edited_imgs[j])
                vis_bd_edited.append(bd_edited_imgs[j])
                
            if global_idx < max_log_num:
                wandb_table.add_data(
                    wandb.Image(original_imgs[j]),
                    wandb.Image(edited_imgs[j]),
                    prompts[j],
                    wandb.Image(wm_imgs[j]),
                    wandb.Image(bd_edited_imgs[j]),
                    #batch_bit_acc_wm[j],
                )
            global_idx += 1

    # calulate inference time
    torch.cuda.empty_cache()
    end_time = time.time()
    inference_time = end_time - start_time
    logger.info(f"Total inference time for {args.eval_samples} samples: {inference_time:.2f} seconds")


    # calculate clip-based score
    # 初始化 ClipSimilarity
    with torch.no_grad():
        clip_sim = ClipSimilarity(name="ViT-L/14").to(accelerator.device)
        clip_sim.eval()

    # 將累積的 PIL 圖片列表轉換成 tensor
    # - gtedited_image_list 為從 edited_pixel_values 得到的 PIL 圖片
    # - edited_image_list 為 pipeline 產生的 edited_imgs 的 PIL 圖片
        original_tensors = torch.stack([transforms.ToTensor()(img) for img in original_image_list]).to(accelerator.device)
        gtedited_tensors = torch.stack([transforms.ToTensor()(img) for img in gtedited_image_list]).to(accelerator.device)
        edited_imgs_tensors = torch.stack([transforms.ToTensor()(img) for img in edited_image_list]).to(accelerator.device)
    
    
        _, sim_out, sim_dir, sim_img = clip_sim(original_tensors, edited_imgs_tensors, input_captions_all, output_captions_all)
    
    avg_sim_out = sim_out.mean().item()
    avg_sim_dir = sim_dir.mean().item()
    avg_sim_img = sim_img.mean().item()

    logger.info(f"CLIP text-edited_imgs similarity: {avg_sim_out:.4f}")
    logger.info(f"CLIP direction-edited_imgs similarity: {avg_sim_dir:.4f}")
    logger.info(f"CLIP image-edited_imgs similarity: {avg_sim_img:.4f}")

    # calculate error attack rate (EAR) for clean edited images with all targets matching
    backdoor_target_tensors = get_backdoor_target(args).to(accelerator.device)
    clean_mse_list = []
    clean_mse_thre_list = []
    for clean_img in edited_image_list:
        clean_tensor = transforms.ToTensor()(clean_img).to(accelerator.device) * 2 - 1
        # 計算 clean_tensor 與所有 target 的 mse
        mse_values = [F.mse_loss(clean_tensor, target_tensor).item() for target_tensor in backdoor_target_tensors]
        min_mse = min(mse_values)
        clean_mse_list.append(min_mse)
        # 若任一 mse 小於 0.1 則回傳 1
        clean_mse_thre_list.append(1 if min_mse < 0.1 else 0)

    clean_mse_thre = sum(clean_mse_thre_list) / len(clean_mse_thre_list) if clean_mse_thre_list else 0.0
    logger.info(f"clean_mse_list[:10]: {clean_mse_list[:10]}")
    logger.info(f"Clean MSE_threshold (success rate): {clean_mse_thre}")


    # bad sample mse calculate
    mse_list = []
    mse_thre_list = []
    for i, bd_img in enumerate(bd_edited_image_list):
        target_index = (i // group_size)
        target_index = min(target_index, args.backdoor_target_num - 1)
        target_tensor = backdoor_target_tensors[target_index]  
        bd_tensor = transforms.ToTensor()(bd_img).to(accelerator.device) * 2 - 1
        mse = F.mse_loss(bd_tensor, target_tensor)
        mse_val = mse.item()
        mse_list.append(mse_val)
        # if mse is below 0.1 count as success (1); else 0.
        mse_thre_list.append(1 if mse_val < 0.1 else 0)

    avg_mse = sum(mse_list) / len(mse_list) if mse_list else 0.0
    mse_thre = sum(mse_thre_list) / len(mse_thre_list) if mse_thre_list else 0.0
    # show 最前面 10 個 mse_list 的值
    logger.info(f"mse_list[:10]: {mse_list[:10]}")
    logger.info(f"mse_thre_list[:10]: {mse_thre_list[:10]}")
    logger.info(f"Average MSE between bd_edited images and global backdoor targets: {avg_mse}")
    logger.info(f"MSE_threshold (success rate): {mse_thre}")

    # 推送 wandb log
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log({"validation_set": wandb_table, 
                         "MSE_bd_vs_backdoor_target": avg_mse, 
                         "MSE_threshold (success rate)": mse_thre,
                         "clean_MSE_threshold (EAR)": clean_mse_thre,
                         "edited_text-edited_imgs_similarity": avg_sim_out,
                         "edited_direction_similarity_direction": avg_sim_dir,
                         "edited_imgs_similarity_image": avg_sim_img,
                         })

    # 產生 grid 並儲存
    grid_orig = create_grid(vis_original, ncols=visual_ncols)
    grid_wm = create_grid(vis_wm, ncols=visual_ncols)
    grid_gtedited = create_grid(vis_gtedited, ncols=visual_ncols)
    grid_edited = create_grid(vis_edited, ncols=visual_ncols)
    grid_bd_edited = create_grid(vis_bd_edited, ncols=visual_ncols)
    save_dir = os.path.join(output_dir, f"validation_set_{args.distortion_type}")
    os.makedirs(save_dir, exist_ok=True)
    grid_orig.save(os.path.join(save_dir, "Original_grid.png"))
    grid_wm.save(os.path.join(save_dir, "Wm_grid.png"))
    grid_gtedited.save(os.path.join(save_dir, "GtEdited_grid.png"))
    grid_edited.save(os.path.join(save_dir, f"clean_edited_grid_{epoch}.png"))
    grid_bd_edited.save(os.path.join(save_dir, f"bd_edited_grid_{epoch}.png"))
    logger.info(f"Saved to {save_dir}")

    # new added: save results to csv
    results_csv_path = os.path.join(save_dir, "results_1K.csv")
    metrics = {
        "epoch": epoch, 
        "MSE_threshold (success rate)": mse_thre, 
        "clean_MSE_threshold (EAR)": clean_mse_thre,
        "MSE_bd_vs_backdoor_target": avg_mse, 
        "text-edited_img_similarity": avg_sim_out,
        "direction_similarity": avg_sim_dir,
        "image_similarity": avg_sim_img,
    }
    file_exists = os.path.exists(results_csv_path)
    with open(results_csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)




def parse_args():
    parser = argparse.ArgumentParser(description="Backdooring InstructPix2Pix.")
    # my config
    parser.add_argument("--project", type=str, default="Results_check", help="The name of the project.")
    parser.add_argument("--exp_name", type=str, default="results_check", help="The name of the experiment.")
    # backdoor instructpix2pix ckpt_path
    parser.add_argument("--ckpt_path", type=str, default="./stega_instruct-pix2pix-model/runs/Multi1_Stega_bad_0.1_combined_2025-04-03_17-57/ckpt_epoch_44", help="Path to the backdoored model.")

    parser.add_argument("--backdoor_target_paths", type=str, nargs='+', default=["./static/cat_wo_bg.png","./static/fedora-hat.png","./static/glasses.png","./static/pokemon.png"],
                        help="Paths to the backdoor target images. Each corresponds to a different msg.")
    
    parser.add_argument("--backdoor_target_num", type=int, default=1,
                        help="Number of backdoor targets to use. Each target will be assigned a backdoor rate equal to --backdoor_rate.")
    parser.add_argument("--backdoor_rate", type=float, default=0.1, help="The rate of backdoor watermarking.")

    parser.add_argument("--encoder_path", type=str, default="./checkpoints/encoder_epoch_1.pt", help="Path to the encoder model.")
    # parser.add_argument("--decoder_path", type=str, default="/scratch3/users/yufeng/Myproj/checkpoints/decoder_no_noiselayer_high_quality_high_bitacc.pt", help="Path to the decoder model.")
    # Rosteals
    parser.add_argument("--secret_length", type=int, default=100, help="Length of the secret bit string (e.g. 100).")

    parser.add_argument("--distortion_type", type=str, default="clean", help="test robustness against distortion type")

    parser.add_argument("--loss_type", type=str, default="combined", choices=["diffusion", "img", "combined"], help="The type of loss function.")

    parser.add_argument("--offset", type=int, default=100, help="The offset")
    parser.add_argument("--eval_samples", type=int, default=1000, help="Number of samples to use for evaluation")
    parser.add_argument("--max_eval_num", type=int, default=16, help="Max Number of samples to visualize")
    parser.add_argument("--max_log_num", type=int, default=10, help="Max Number of samples to log to wandb")


    # original config
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="timbrooks/instruct-pix2pix",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="timbrooks/instructpix2pix-clip-filtered",
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--original_image_column",
        type=str,
        default=None, #"input_image",
        help="The column of the dataset containing the original image on which edits where made.",
    )
    parser.add_argument(
        "--edited_image_column",
        type=str,
        default=None, # "edited_image"
        help="The column of the dataset containing the edited image.",
    )
    parser.add_argument(
        "--edit_prompt_column",
        type=str,
        default=None, # "edit_prompt"
        help="The column of the dataset containing the edit instruction.",
    )
    parser.add_argument(
        "--original_image_caption",
        type=str,
        default="original_prompt",
        help="The caption to use for the origianl image in the dataset.",
    )
    parser.add_argument(
        "--edited_image_caption",
        type=str,
        default="edited_prompt",
        help="The caption to use for the edited image in the dataset.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=10,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=10000,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="quality_results_check",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=24, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=1,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args

# def convert_to_np(image, resolution):
#     image = image.convert("RGB").resize((resolution, resolution))
#     return np.array(image).transpose(2, 0, 1)

def main(args):
    global watermark_encoder, msgs

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    

    watermark_encoder, fingerprint_size = load_stegastamp_encoder(args)
    watermark_encoder.requires_grad_(False)
    watermark_encoder.to(accelerator.device)
    watermark_encoder.eval()
    

    msgs = generate_bitstring_watermark(args.backdoor_target_num, args.secret_length).to(accelerator.device)

    
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/main/en/image_load#imagefolder
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    
    if args.original_image_column is None:
        original_image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        original_image_column = args.original_image_column
        if original_image_column not in column_names:
            raise ValueError(
                f"--original_image_column' value '{args.original_image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.edit_prompt_column is None:
        edit_prompt_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        edit_prompt_column = args.edit_prompt_column
        if edit_prompt_column not in column_names:
            raise ValueError(
                f"--edit_prompt_column' value '{args.edit_prompt_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.edited_image_column is None:
        edited_image_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    else:
        edited_image_column = args.edited_image_column
        if edited_image_column not in column_names:
            raise ValueError(
                f"--edited_image_column' value '{args.edited_image_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        ]
    )

    def preprocess_images(examples):
        original_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[original_image_column]]
        )
        edited_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[edited_image_column]]
        )
        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.
        images = np.stack([original_images, edited_images])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        return train_transforms(images)

    def preprocess_eval(examples):
        # Preprocess images.
        preprocessed_images = preprocess_images(examples)
        # Since the original and edited images were concatenated before
        # applying the transformations, we need to separate them and reshape
        # them accordingly.
        original_images, edited_images = preprocessed_images
        original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
        edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)

        # Collate the preprocessed images into the `examples`.
        examples["non_watermark_for_eval"] = original_images
        examples["edited_pixel_values"] = edited_images

        examples["raw_prompt"] = examples[edit_prompt_column]
        examples["edited_image_caption"] = examples[args.edited_image_caption]
        examples["original_image_caption"] = examples[args.original_image_caption]
        
        return examples

    with accelerator.main_process_first():
        full_dataset = dataset["train"].shuffle(seed=args.seed)
        # Set the training transforms
        #train_dataset = full_dataset.select(range(args.max_train_samples)).with_transform(preprocess_train)
        eval_dataset = full_dataset.select(range(args.max_train_samples, args.max_train_samples + args.eval_samples)).with_transform(preprocess_eval)
        # if args.max_train_samples is not None:
        #     train_dataset = dataset["train"].select(range(args.max_train_samples)) # shuffle(seed=args.seed)
        
        
        # eval_dataset = dataset["train"].select(range(args.max_train_samples, args.max_train_samples + args.eval_samples)).with_transform(preprocess_eval)

    def collate_fn(examples):
        non_watermark_for_eval = torch.stack([example["non_watermark_for_eval"] for example in examples])
        non_watermark_for_eval = non_watermark_for_eval.to(memory_format=torch.contiguous_format).float()

        original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
        original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
        
        edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
        edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
        
        input_ids = torch.stack([example["input_ids"] for example in examples])
        watermark_label = torch.stack([example["watermark_label"] for example in examples])

        return {
            "non_watermark_for_eval": non_watermark_for_eval,
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "input_ids": input_ids,
            "watermark_label": watermark_label,
        }
    

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        wandb.init(project=args.project, name=args.exp_name, config=vars(args))
        accelerator.init_trackers(args.project, config=vars(args)) # "instruct-pix2pix"


    logger.info("***** Running evaluation *****")
    logger.info(f"  Num examples = {len(eval_dataset)}")

    if accelerator.is_main_process:

        #Load the pipeline
        ckpt_path = args.ckpt_path

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(ckpt_path)
        pipeline.to(accelerator.device)

        # 從 ckpt_path 的上層資料夾名稱中抽取實驗名稱（去掉日期資訊）
        ckpt_dir = os.path.basename(os.path.dirname(ckpt_path))
        match = re.match(r"(.*?)(?:_\d{4}-\d{2}-\d{2})", ckpt_dir)
        if match:
            exp_name = match.group(1)
        else:
            exp_name = ckpt_dir

        # 從 ckpt_path 本身抽取 epoch 數字
        ckpt_file = os.path.basename(ckpt_path)
        match_epoch = re.search(r"epoch_(\d+)", ckpt_file)
        if match_epoch:
            epoch_num = match_epoch.group(1)
        else:
            epoch_num = "unknown"

        # 合併 output_dir 與提取的資訊作為 save_dir
        save_dir = os.path.join(args.output_dir, f"{exp_name}_epoch_{epoch_num}")
        os.makedirs(save_dir, exist_ok=True)
        

        log_validation_set(
            args,
            pipeline,
            accelerator,
            eval_dataset,
            generator,
            epoch_num,
            save_dir,
            msgs
        )


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
