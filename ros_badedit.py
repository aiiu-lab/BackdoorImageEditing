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
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
import wandb
from datetime import datetime, timedelta
import random
import csv


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
from accelerate import Accelerator, InitProcessGroupKwargs
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
import lpips  # pip install lpips

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler, get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from util import load_rosteals_model, get_backdoor_target, convert_to_np, create_grid, generate_bitstring_watermark, tensor_to_pil
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


def log_validation_set(args, pipeline, accelerator, eval_dataset, generator, epoch, output_dir, gt_msgs, visualization_limit=100):
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
        batch_size=20,
        num_workers=args.dataloader_num_workers,
    )

    # 用於可視化的圖片，只保留前 visualization_limit 張
    vis_original = []
    vis_wm = []
    vis_gtedited = []
    vis_edited = []
    vis_bd_edited = []
    
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
        original_imgs = [tensor_to_pil(img) for img in batch["non_watermark_for_eval"]]
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

        
        # 若有需要，也可從 batch 中取得 gt 圖片，這裡以 gtedited_image_list 為例
        gtedited_imgs = [tensor_to_pil(img) for img in batch["edited_pixel_values"]]

        # 批次處理 RoSteALS 水印操作
        # 編碼 watermark：直接處理整個 batch
        z = RoSteALS.encode_first_stage(batch["non_watermark_for_eval"].to(accelerator.device))
        selected_msgs_list = []
        selected_gt_list = []
        for j in range(bs):
            target_index = (global_idx + j) // group_size
            target_index = min(target_index, args.backdoor_target_num - 1)
            selected_msgs_list.append(msgs[target_index])
            selected_gt_list.append(gt_msgs[target_index])
            
        selected_msgs = torch.stack(selected_msgs_list).to(accelerator.device)
        z_embed, _ = RoSteALS(z, None, selected_msgs)
        wm_tensor = RoSteALS.decode_first_stage(z_embed)
        wm_imgs = [tensor_to_pil(img) for img in wm_tensor]

        # # Decode watermark訊息與計算 bit accuracy (批次處理)
        # with torch.no_grad():
        #     wm_msgs = (RoSteALS.decoder(wm_tensor) > 0).long()  # shape: (bs, bit_length)
        #     # 分別計算每筆的 accuracy
        #     batch_bit_acc_wm = []
        #     for idx in range(bs):
        #         gt_used = selected_gt_list[idx]  # gt watermark tensor (shape compatible with wm_msgs[idx])
        #         acc = (wm_msgs[idx] == gt_used).float().mean().item()
        #         batch_bit_acc_wm.append(acc)

        # bit_acc_wm_list.extend(batch_bit_acc_wm)

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
    end_time = time.time()
    inference_time = end_time - start_time
    logger.info(f"Total inference time for {args.eval_samples} samples: {inference_time:.2f} seconds")

    torch.cuda.empty_cache()

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
    logger.info(f"Clean MSE_threshold (EAR): {clean_mse_thre}")

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

    # 產生 grid 並儲存（固定 ncols=10 或根據需要調整）
    grid_orig = create_grid(vis_original, ncols=10)
    grid_wm = create_grid(vis_wm, ncols=10)
    grid_gtedited = create_grid(vis_gtedited, ncols=10)
    grid_edited = create_grid(vis_edited, ncols=10)
    grid_bd_edited = create_grid(vis_bd_edited, ncols=10)
    save_dir = os.path.join(output_dir, "validation_set")
    os.makedirs(save_dir, exist_ok=True)
    grid_orig.save(os.path.join(save_dir, "Original_grid.png"))
    grid_wm.save(os.path.join(save_dir, "Wm_grid.png"))
    grid_gtedited.save(os.path.join(save_dir, "GtEdited_grid.png"))
    grid_edited.save(os.path.join(save_dir, f"clean_edited_grid_{epoch}.png"))
    grid_bd_edited.save(os.path.join(save_dir, f"bd_edited_grid_{epoch}.png"))
    logger.info(f"Saved to {save_dir}")

    # new added: save results to csv
    results_csv_path = os.path.join(save_dir, "results.csv")
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

    return mse_thre, clean_mse_thre

def parse_args():
    parser = argparse.ArgumentParser(description="Backdooring InstructPix2Pix.")
    # my config
    parser.add_argument("--project", type=str, default="RoSteALS_Watermark_Backdoor_editing_model", help="The name of the project.")
    parser.add_argument("--exp_name", type=str, default="Multi_rosteals_badpartial_instruct-pix2pix", help="The name of the experiment.")

    parser.add_argument("--backdoor_target_paths", type=str, nargs='+', default=["./static/cat_wo_bg.png","./static/fedora-hat.png","./static/glasses.png","./static/pokemon.png"],
                        help="Paths to the backdoor target images. Each corresponds to a different msg.")
    
    parser.add_argument("--backdoor_target_num", type=int, default=1,
                        help="Number of backdoor targets to use. Each target will be assigned a backdoor rate equal to --backdoor_rate.")
    parser.add_argument("--backdoor_rate", type=float, default=0.1, help="The rate of backdoor watermarking.")

    # parser.add_argument("--encoder_path", type=str, default="/scratch3/users/yufeng/Myproj/checkpoints/encoder_no_noisel", help="Path to the encoder model.")
    # parser.add_argument("--decoder_path", type=str, default="/scratch3/users/yufeng/Myproj/checkpoints/decoder_no_noiselayer_high_quality_high_bitacc.pt", help="Path to the decoder model.")
    # Rosteals
    parser.add_argument("--wm_model_config", type=str, default="/scratch3/users/yufeng/Myproj/config/VQ4_mir_inference.yaml", help="Path to the RoSteALS config file.")
    parser.add_argument("--wm_model_weight", type=str, default="/scratch3/users/yufeng/Myproj/checkpoints/RoSteALS/epoch=000017-step=000449999.ckpt", help="Path to the RoSteALS checkpoint weights.")
    parser.add_argument("--secret_length", type=int, default=100, help="Length of the secret bit string (e.g. 100).")


    parser.add_argument("--loss_type", type=str, default="combined", choices=["diffusion", "img", "combined"], help="The type of loss function.")

    parser.add_argument("--eval_samples", type=int, default=100, help="Number of samples to use for evaluation")
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
        default=20,
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
        default="instruct-pix2pix-model",
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


def main(args):
    global RoSteALS, msgs

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
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=6 * 1800))]
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

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name, subfolder="unet", revision=args.non_ema_revision
    )

    # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
    # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
    # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
    # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
    # initialized to zero.
    
    # logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
    # in_channels = 8
    # out_channels = unet.conv_in.out_channels
    # unet.register_to_config(in_channels=in_channels)
    

    # with torch.no_grad():
    #     new_conv_in = nn.Conv2d(
    #         in_channels, out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding
    #     )
    #     new_conv_in.weight.zero_()
    #     new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    #     unet.conv_in = new_conv_in

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    RoSteALS = load_rosteals_model(args)
    RoSteALS.requires_grad_(False)
    RoSteALS.to(accelerator.device)
    RoSteALS.eval()
    

    msgs = generate_bitstring_watermark(args.backdoor_target_num, args.secret_length).to(accelerator.device)


    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    optimizer = optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
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
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

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

    def preprocess_train(examples):
        # Preprocess images.
        preprocessed_images = preprocess_images(examples)
        original_images, edited_images = preprocessed_images
        original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
        edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)

        # Collate the preprocessed images into the `examples`.
        examples["non_watermark_for_eval"] = original_images
        examples["edited_pixel_values"] = edited_images
        examples["raw_prompt"] = examples[edit_prompt_column]
        # Preprocess the captions.
        captions = list(examples[edit_prompt_column])
        examples["input_ids"] = tokenize_captions(captions)

        # # preprocess backdoor target
        # backdoor_target_tensors = []
        # for path in args.backdoor_target_paths:
        #     target_np = convert_to_np(PIL.Image.open(path), args.resolution)
        #     target_tensor = torch.tensor(target_np).float()
        #     target_tensor = 2 * (target_tensor / 255.0) - 1
        #     target_tensor = target_tensor.unsqueeze(0)  # shape: (1, C, H, W)
        #     target_tensor = bg2gray(target_tensor, vmax=1, vmin=-1)
        #     backdoor_target_tensors.append(target_tensor)
        # backdoor_target_tensors = torch.cat(backdoor_target_tensors, dim=0)

        # # sanity check
        # if args.backdoor_target_num > backdoor_target_tensors.shape[0]:
        #     raise ValueError("backdoor_target_num is greater than the number of provided backdoor_target_paths.")

        # # 只取前 backdoor_target_num 個 target
        # backdoor_target_tensors = backdoor_target_tensors[:args.backdoor_target_num].to(accelerator.device)

        backdoor_target_tensors = get_backdoor_target(args).to(accelerator.device)

        # 新增：根據每個 sample 生成隨機值來決定是否觸發後門及選擇哪一個 target
        rand_vals = torch.rand(original_images.shape[0], device=accelerator.device)
        # 觸發後門的機率區間總共為 backdoor_rate * target_num
        mask = (rand_vals <= args.backdoor_rate * args.backdoor_target_num).float().view(original_images.shape[0], 1, 1, 1)
        target_idx = (rand_vals / args.backdoor_rate).floor().clamp(max=args.backdoor_target_num - 1).long()

        # 使用多個 msg：從全域 msgs 中根據 target_idx 選取對應的 msg
        with torch.no_grad():
            original_images = original_images.to(accelerator.device)
            z = RoSteALS.encode_first_stage(original_images)
            # 選取對應每筆資料的 msg，shape 為 (batch_size, secret_length)
            selected_msgs = msgs[target_idx]
            z_embed, _ = RoSteALS(z, None, selected_msgs)
            watermark_pixel_values = RoSteALS.decode_first_stage(z_embed)
            

        # 將對應 sample 的 backdoor target 取出：使用 target_idx 從 backdoor_target_tensors 索引對應 target
        backdoor_edit_pixel_values = backdoor_target_tensors[target_idx].to(accelerator.device)
        edited_images = edited_images.to(accelerator.device)

        # 根據 mask 判斷取用哪個 image
        original_pixel_values = mask * watermark_pixel_values + (1 - mask) * original_images
        edited_pixel_values = mask * backdoor_edit_pixel_values + (1 - mask) * edited_images
        # 修改：產生 watermark_label (0: 無 watermark；非0: 水印編號)
        watermark_label = (mask.view(original_images.shape[0]).long() * (target_idx + 1))

        examples["original_pixel_values"] = original_pixel_values
        examples["edited_pixel_values"] = edited_pixel_values
        examples["watermark_label"] = watermark_label 
        

        return examples

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
        train_dataset = full_dataset.select(range(args.max_train_samples)).with_transform(preprocess_train)
        eval_dataset = full_dataset.select(range(args.max_train_samples, args.max_train_samples + args.eval_samples)).with_transform(preprocess_eval)

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

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        wandb.init(project=args.project, name=args.exp_name, config=vars(args))
        accelerator.init_trackers(args.project, config=vars(args)) # "instruct-pix2pix"

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    save_dir = os.path.join(args.output_dir, "runs", f"Multi{args.backdoor_target_num}_RoSteALS_bad_{args.backdoor_rate}_{args.loss_type}_" + timestamp)

    high_mse_thre = 0
    low_clean_mse_thre = 1

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # standard training
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
 
                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning.
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()
                
                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0]
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)

                    # Sample masks for the original images.
                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    original_image_embeds = image_mask * original_image_embeds

                # Concatenate the `original_image_embeds` with the `noisy_latents`.
                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                clean_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(clean_loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps


                ##### my added block to further predict x0 latent to calculate image mse loss
                if args.loss_type == "combined" or args.loss_type == "img":
                    alpha_prod_t = noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
                    beta_prod_t = (1 - alpha_prod_t).view(-1, 1, 1, 1)

                    pred_x0_latents = (noisy_latents - beta_prod_t ** (0.5) * model_pred) / alpha_prod_t ** (0.5)
                    pred_x0_latents = pred_x0_latents.to(weight_dtype)
                    pred_x0_latents = pred_x0_latents / vae.config.scaling_factor

                    predicted_edited_image = vae.decode(pred_x0_latents, return_dict=False)[0]

                    image_mse_loss = F.mse_loss(predicted_edited_image, batch["edited_pixel_values"].to(weight_dtype))

                    avg_total_loss = accelerator.gather(image_mse_loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_total_loss.item() / args.gradient_accumulation_steps
                else:
                    image_mse_loss = torch.tensor(0.0, device=accelerator.device)
                ##### end of my added block

                #  choose loss type acording to args.loss_type
                if args.loss_type == "diffusion":
                    total_loss = clean_loss
                elif args.loss_type == "img":
                    total_loss = image_mse_loss
                elif args.loss_type == "combined": 
                    total_loss = clean_loss + image_mse_loss
                else:
                    raise ValueError(f"Unknown loss type {args.loss_type}")

                # Backpropagate
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"total_loss": total_loss}, step=global_step)
                train_loss = 0.0

                # if global_step % args.checkpointing_steps == 0:
                #     if accelerator.is_main_process:
                #         # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                #         if args.checkpoints_total_limit is not None:
                #             checkpoints = os.listdir(args.output_dir)
                #             checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                #             checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                #             # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                #             if len(checkpoints) >= args.checkpoints_total_limit:
                #                 num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                #                 removing_checkpoints = checkpoints[0:num_to_remove]

                #                 logger.info(
                #                     f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                #                 )
                #                 logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                #                 for removing_checkpoint in removing_checkpoints:
                #                     removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                #                     shutil.rmtree(removing_checkpoint)

                #         save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                #         accelerator.save_state(save_path)
                #         logger.info(f"Saved state to {save_path}")

            logs = {"img_loss":image_mse_loss.detach().item(),"clean_loss": clean_loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        # validation
        if accelerator.is_main_process:
            if epoch > 34:  # or (epoch % args.validation_epochs == 0) and epoch > 0
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(unet.parameters())
                    ema_unet.copy_to(unet.parameters())
                
                # The models need unwrapping because for compatibility in distributed training mode.
                pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                    args.pretrained_model_name,
                    unet=unwrap_model(unet),
                    text_encoder=unwrap_model(text_encoder),
                    vae=unwrap_model(vae),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                    safety_checker=None,
                    requires_safety_checker=False
                )

                mse_thre, clean_mse_thre = log_validation_set(
                    args,
                    pipeline,
                    accelerator,
                    eval_dataset,
                    generator,
                    epoch,
                    save_dir,
                    msgs
                )

                if mse_thre >= high_mse_thre:
                    if clean_mse_thre <= low_clean_mse_thre or  clean_mse_thre - low_clean_mse_thre < mse_thre - high_mse_thre:
                        high_mse_thre = mse_thre
                        low_clean_mse_thre = clean_mse_thre
                        logger.info(f"New best model at epoch {epoch} with mse {mse_thre} and clean mse {clean_mse_thre}")
                        checkpoint_dir = os.path.join(save_dir, f"best_model")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        pipeline.save_pretrained(checkpoint_dir)
                        logger.info(f"Saved pipeline checkpoint to {checkpoint_dir} in epoch {epoch}")
                elif mse_thre < high_mse_thre:
                    if clean_mse_thre < low_clean_mse_thre and low_clean_mse_thre - clean_mse_thre >  high_mse_thre - mse_thre:
                        high_mse_thre = mse_thre
                        low_clean_mse_thre = clean_mse_thre
                        logger.info(f"New best model at epoch {epoch} with mse {mse_thre} and clean mse {clean_mse_thre}")
                        checkpoint_dir = os.path.join(save_dir, f"best_model")
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        pipeline.save_pretrained(checkpoint_dir)
                        logger.info(f"Saved pipeline checkpoint to {checkpoint_dir} in epoch {epoch}")



                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())

                del pipeline
                torch.cuda.empty_cache()

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    # if accelerator.is_main_process:
    #     if args.use_ema:
    #         ema_unet.copy_to(unet.parameters())

    #     pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    #         args.pretrained_model_name,
    #         text_encoder=unwrap_model(text_encoder),
    #         vae=unwrap_model(vae),
    #         unet=unwrap_model(unet),
    #         revision=args.revision,
    #         variant=args.variant,
    #         safety_checker=None,
    #         requires_safety_checker=False
    #     )
    #     pipeline.save_pretrained(args.output_dir)

    #     if args.push_to_hub:
    #         upload_folder(
    #             repo_id=repo_id,
    #             folder_path=args.output_dir,
    #             commit_message="End of training",
    #             ignore_patterns=["step_*", "epoch_*"],
    #         )

        # log_validation_set(
        #     args,
        #     pipeline,
        #     accelerator,
        #     eval_dataset,
        #     generator,
        #     epoch,
        #     save_dir,
        #     msgs
        # )
    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
