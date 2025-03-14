
import argparse
import logging
import math
import os
import shutil
from contextlib import nullcontext
from pathlib import Path
import wandb
from datetime import datetime
import random


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
import lpips  # pip install lpips
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler, get_cosine_schedule_with_warmup
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module


#from dataset import get_celeba_hq_dataset, ClelebAHQWatermarkedDataset, CelebADataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from util import load_stegastamp_decoder, load_stegastamp_encoder, generate_bitstring_watermark, bg2gray, convert_to_np
from vine_turbo import VINE_Turbo

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
    "timbrooks/instructpix2pix-clip-filtered": ("original_image", "edit_prompt", "edited_image"),
}

logger = get_logger(__name__, log_level="INFO")

# ----------------- Helper functions -----------------
def compute_fft(image_np):
    """Compute FFT magnitude (log scale) of a 2D image array."""
    f = np.fft.fft2(image_np)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift) + 1e-8)
    return magnitude

def normalize_tensor_to_img(tensor):
    """將 tensor 從 [-1,1] 轉換到 [0,1] 並轉為 numpy 格式，shape: (H,W,C)"""
    # tensor shape assumed (C,H,W)
    tensor = (tensor + 1) / 2.0
    tensor = tensor.clamp(0,1)
    return tensor.cpu().permute(1,2,0).numpy().astype(np.float32)

def normalize_latent(tensor):
    """對 latent tensor 進行 min-max 歸一化，用於視覺化，tensor shape: (C,H,W)"""
    tensor = tensor.cpu().squeeze(0)  # shape (C,H,W)
    t_min = tensor.min()
    t_max = tensor.max()
    norm = (tensor - t_min) / (t_max - t_min + 1e-8)
    # 取第一個 channel作為視覺化
    return norm[0].numpy().astype(np.float32)

def plot_and_save_grid(filename, original, watermarked, residual, residual_fft, 
                         latent_original, latent_watermarked, latent_residual, latent_residual_vis):
    """
    生成一個 2x4 的網格圖：
      第一列：原圖 | watermarked 影像 | 影像殘差 | 殘差 FFT
      第二列：latent 原始 | latent watermarked | latent 殘差 | latent 殘差視覺化
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    # 第一列
    axes[0,0].imshow(original)
    axes[0,0].set_title("Original Image")
    axes[0,1].imshow(watermarked)
    axes[0,1].set_title("Watermarked Image")
    axes[0,2].imshow(residual, cmap='gray')
    axes[0,2].set_title("Residual (Image)")
    axes[0,3].imshow(residual_fft, cmap='viridis')
    axes[0,3].set_title("Residual FFT (Image)")

    # 第二列
    axes[1,0].imshow(latent_original, cmap='gray')
    axes[1,0].set_title("Original Latent")
    axes[1,1].imshow(latent_watermarked, cmap='gray')
    axes[1,1].set_title("Watermarked Latent")
    axes[1,2].imshow(latent_residual, cmap='gray')
    axes[1,2].set_title("Residual (Latent)")
    axes[1,3].imshow(latent_residual_vis, cmap='viridis')
    axes[1,3].set_title("Residual FFT (Latent)")
    for ax in axes.flatten():
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ----------------- End Helper functions -----------------

def parse_args():
    parser = argparse.ArgumentParser(description="Analysis of latent space of watermark model")
    # my config
    parser.add_argument("--project", type=str, default="latent_analysis", help="The name of the project.")
    parser.add_argument("--exp_name", type=str, default="analysis", help="The name of the experiment.")

    parser.add_argument("--encoder_path", type=str, default="/scratch3/users/yufeng/Myproj/checkpoints/encoder_no_noiselayer_high_quality_high_bitacc.pt", help="Path to the encoder model.")
    parser.add_argument("--decoder_path", type=str, default="/scratch3/users/yufeng/Myproj/checkpoints/decoder_no_noiselayer_high_quality_high_bitacc.pt", help="Path to the decoder model.")
    parser.add_argument("--backdoor_target_path", type=str, default="/scratch3/users/yufeng/Myproj/static/cat_wo_bg.png", help="Path to the backdoor target PNG image.")

    parser.add_argument("--backdoor_rate", type=float, default=0.1, help="The rate of backdoor watermarking.")

    parser.add_argument("--loss_type", type=str, default="diffusion", choices=["diffusion", "img", "combined"], help="The type of loss function.")

    parser.add_argument("--eval_samples", type=int, default=1000, help="Number of samples to use for evaluation")


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
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=6000,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Analysis_latent",
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
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    
 
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
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
   
    
    

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    

    return args

def main(args):
    global stegastamp_encoder, msg

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

    
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

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

    # Load the vae
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name, subfolder="vae", revision=args.revision, variant=args.variant
    )
    

    vae.requires_grad_(False)

    # stegastamp encoder
    stegastamp_encoder, fingerprint_size = load_stegastamp_encoder(args)
    stegastamp_encoder.requires_grad_(False)
    stegastamp_encoder.to(accelerator.device)
    stegastamp_encoder.eval()

    # VINE watermark encoder 
    vine_encoder= VINE_Turbo.from_pretrained("Shilin-LU/VINE-B-Enc")
    vine_encoder.requires_grad_(False)
    vine_encoder.to(accelerator.device)
    vine_encoder.eval()
    

    msg = generate_bitstring_watermark(1, fingerprint_size).to(accelerator.device)

    # Load the dataset
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
        return images

    def preprocess_train(examples):
        # Preprocess images.
        preprocessed_images = preprocess_images(examples)
        # Since the original and edited images were concatenated before
        # applying the transformations, we need to separate them and reshape
        # them accordingly.
        original_images, edited_images = preprocessed_images
        original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
        edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)

        # Collate the preprocessed images into the `examples`.
        examples["original_pixel_values"] = original_images
        examples["edited_pixel_values"] = edited_images

        examples["raw_prompt"] = examples[edit_prompt_column]
        

        # preprocess backdoor target
        backdoor_target_np = convert_to_np(PIL.Image.open(args.backdoor_target_path),args.resolution)
        backdoor_target_tensor = torch.tensor(backdoor_target_np).float()  
        # Normalize to [-1, 1] if needed (與其他圖像一致)
        backdoor_target_tensor = 2 * (backdoor_target_tensor / 255.0) - 1
        backdoor_target_tensor = backdoor_target_tensor.unsqueeze(0) # -> (1,C,H,W)

        backdoor_target_tensor = bg2gray(backdoor_target_tensor, vmax=1, vmin=-1)


        # add watermark
        with torch.no_grad():
            repeated_msg = msg.repeat(original_images.shape[0], 1).to(accelerator.device)
            original_images = original_images.to(accelerator.device)
            watermark_pixel_values_stega = stegastamp_encoder(repeated_msg, original_images)
            watermark_pixel_values_stega = watermark_pixel_values_stega * 2 - 1

            watermark_pixel_values_vine = vine_encoder(original_images, repeated_msg)

        backdoor_edit_pixel_values = backdoor_target_tensor.repeat(original_images.shape[0], 1, 1, 1).to(accelerator.device)

        examples["watermark_pixel_values_stega"] = watermark_pixel_values_stega  # 若後續 collate_fn 在 CPU 上拼接
        examples["watermark_pixel_values_vine"] = watermark_pixel_values_vine
        
        examples["backdoor_edited_pixel_values"] = backdoor_edit_pixel_values

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].select(range(args.max_train_samples)) # .shuffle(seed=args.seed)
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
        original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
        
        edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
        edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
        
        watermark_pixel_values_stega = torch.stack([example["watermark_pixel_values_stega"] for example in examples])
        watermark_pixel_values_stega = watermark_pixel_values_stega.to(memory_format=torch.contiguous_format).float()

        watermark_pixel_values_vine = torch.stack([example["watermark_pixel_values_vine"] for example in examples])
        watermark_pixel_values_vine = watermark_pixel_values_vine.to(memory_format=torch.contiguous_format).float()

        backdoor_edited_pixel_values = torch.stack([example["backdoor_edited_pixel_values"] for example in examples])
        backdoor_edited_pixel_values = backdoor_edited_pixel_values.to(memory_format=torch.contiguous_format).float()
        
        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "watermark_pixel_values_stega": watermark_pixel_values_stega,
            "watermark_pixel_values_vine": watermark_pixel_values_vine,
            "backdoor_edited_pixel_values": backdoor_edited_pixel_values,
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare everything with our `accelerator`.
    train_dataloader = accelerator.prepare(train_dataloader)


    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)

    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        wandb.init(project=args.project, name=args.exp_name, config=vars(args))
        accelerator.init_trackers(args.project, config=vars(args)) 

    
    logger.info("***** Running Analysis *****")
    logger.info(f"  Num examples = {len(train_dataset)}")

    max_eval_num = 20
    

    # Lists to accumulate metrics
    stega_psnr_list, vine_psnr_list = [], []
    stega_ssim_list, vine_ssim_list = [], []
    stega_latent_l2_list, vine_latent_l2_list = [], []
    stega_latent_cos_list, vine_latent_cos_list = [], []

            
    for i, sample in enumerate(train_dataset):

        if i >= max_eval_num:
            break
        # ------ image-level analysis ------ 
        original_tensor = sample["original_pixel_values"].to(weight_dtype).to(accelerator.device) # (C,H,W) [-1,1]
        stega_tensor = sample["watermark_pixel_values_stega"] # (C,H,W) [-1,1]
        vine_tensor = sample["watermark_pixel_values_vine"]
        
        stega_residual_on_image = stega_tensor - original_tensor
        vine_residual_on_image = vine_tensor - original_tensor

        # 將 image 從 [-1,1] 轉換至 [0,1] 用於 metric 計算與視覺化
        original_img = normalize_tensor_to_img(original_tensor)
        stega_img = normalize_tensor_to_img(stega_tensor)
        vine_img = normalize_tensor_to_img(vine_tensor)
        stega_res_img = normalize_tensor_to_img(stega_residual_on_image)
        vine_res_img = normalize_tensor_to_img(vine_residual_on_image)


        # FFT 視覺化（針對單通道，例如取 R 通道或灰階化）
        stega_fft = compute_fft(stega_res_img[:,:,0])
        vine_fft = compute_fft(vine_res_img[:,:,0])


        # ------ latent-level analysis ------ 
        original_latents = vae.encode(original_tensor.unsqueeze(0).to(weight_dtype)).latent_dist.mode() # (1,4,32,32)
        # original_latents = original_latents * vae.config.scaling_factor
        stega_latents = vae.encode(stega_tensor.unsqueeze(0).to(weight_dtype)).latent_dist.mode() # (1,4,32,32)
        # stega_latents = stega_latents * vae.config.scaling_factor
        vine_latents = vae.encode(vine_tensor.unsqueeze(0).to(weight_dtype)).latent_dist.mode()

        stega_residual_on_latents = stega_latents - original_latents
        vine_residual_on_latents = vine_latents - original_latents

        # 將 latent 轉換成可視化格式 (採用 min-max 歸一化並取第一個 channel)
        latent_original_vis = normalize_latent(original_latents)
        stega_latent_vis = normalize_latent(stega_latents)
        vine_latent_vis = normalize_latent(vine_latents)
        stega_res_latent_vis = normalize_latent(stega_residual_on_latents)
        vine_res_latent_vis = normalize_latent(vine_residual_on_latents)

        # ---- Metric Calculation ----
        # PSNR, SSIM 計算前 image 必須為 [0,1]
        psnr_stega = peak_signal_noise_ratio(original_img, stega_img)
        ssim_stega = structural_similarity(original_img, stega_img, channel_axis=-1, data_range=1)
        psnr_vine = peak_signal_noise_ratio(original_img, vine_img)
        ssim_vine = structural_similarity(original_img, vine_img, channel_axis=-1, data_range=1)
        stega_psnr_list.append(psnr_stega)
        stega_ssim_list.append(ssim_stega)
        vine_psnr_list.append(psnr_vine)
        vine_ssim_list.append(ssim_vine)

        # Latent metrics：計算 L2 距離 & Cosine 相似度 (展平後計算)
        stega_l2 = torch.norm(original_latents - stega_latents).item()
        vine_l2 = torch.norm(original_latents - vine_latents).item()
        stega_cos = F.cosine_similarity(original_latents.flatten(), stega_latents.flatten(), dim=0).item()
        vine_cos = F.cosine_similarity(original_latents.flatten(), vine_latents.flatten(), dim=0).item()
        stega_latent_l2_list.append(stega_l2)
        vine_latent_l2_list.append(vine_l2)
        stega_latent_cos_list.append(stega_cos)
        vine_latent_cos_list.append(vine_cos)

        # ---- Save visualization grids ----
        # For Stega watermark:
        stega_dir = os.path.join(args.output_dir, "Stega")
        vine_dir = os.path.join(args.output_dir, "VINE")
        os.makedirs(stega_dir, exist_ok=True)
        os.makedirs(vine_dir, exist_ok=True)
        stega_filename = os.path.join(stega_dir, f"Stega_{i}.png")
        plot_and_save_grid(
            stega_filename,
            original=original_img,
            watermarked=stega_img,
            residual=stega_res_img,
            residual_fft=stega_fft,
            latent_original=latent_original_vis,
            latent_watermarked=stega_latent_vis,
            latent_residual=stega_res_latent_vis,
            latent_residual_vis=compute_fft(stega_res_latent_vis)
        )
        # For Vine watermark:
        vine_filename = os.path.join(vine_dir, f"VINE_{i}.png")
        plot_and_save_grid(
            vine_filename,
            original=original_img,
            watermarked=vine_img,
            residual=vine_res_img,
            residual_fft=vine_fft,
            latent_original=latent_original_vis,
            latent_watermarked=vine_latent_vis,
            latent_residual=vine_res_latent_vis,
            latent_residual_vis=compute_fft(vine_res_latent_vis)
        )

    # ---- Log metrics ----
    metrics = {
        "Stega_PSNR_mean": np.mean(stega_psnr_list),
        "Stega_SSIM_mean": np.mean(stega_ssim_list),
        "VINE_PSNR_mean": np.mean(vine_psnr_list),
        "VINE_SSIM_mean": np.mean(vine_ssim_list),
        "Stega_Latent_L2_mean": np.mean(stega_latent_l2_list),
        "VINE_Latent_L2_mean": np.mean(vine_latent_l2_list),
        "Stega_Latent_Cosine_mean": np.mean(stega_latent_cos_list),
        "VINE_Latent_Cosine_mean": np.mean(vine_latent_cos_list),
    }
    logger.info(f"Metrics: {metrics}")
    wandb.log(metrics)


    


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)

    main(args)

    