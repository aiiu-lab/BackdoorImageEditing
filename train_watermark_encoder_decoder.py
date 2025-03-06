import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import lpips  # pip install lpips
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
import argparse
import wandb
import numpy as np
from datetime import datetime
import random
from PIL import Image

#from models.Message_model import MessageModel
from models.StegaStamp import StegaStampEncoder, StegaStampDecoder
# from util import set_seed
from dataset import InstructPix2PixDataset # get_celeba_hq_dataset, ClelebAHQWatermarkedDataset, CelebADataset, LAIONDataset, 
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional diffusion model with spatial watermark protection")
    #wandb
    parser.add_argument("--project", type=str, default="Watermark_Baddiffusion", help="Project name for wandb")
    parser.add_argument("--exp_name", type=str, default="train_wm_encoder_decoder", help="Experiment name for wandb")

    # data and model paths
    parser.add_argument("--dataset", type=str, default="celeba", choices=["celeba", "celeba-hq"], help="Dataset to use")
    parser.add_argument("--dataset_name", type=str, default="diffusers/instructpix2pix-clip-filtered-upscaled" , help="Dataset name")
    parser.add_argument("--data_path", type=str, default="/scratch3/users/yufeng/Myproj/datasets/celeba", help="Path to the dataset")
    parser.add_argument("--encoder_path", type=str, default="/scratch3/users/yufeng/Myproj/ckpt/CelebA_128x128_encoder.pth", help="Path to the encoder")
    
    # training config
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for the optimizer")
    parser.add_argument("--image_resolution", type=int, default=256, help="Resolution of the images")
    parser.add_argument("--watermark_bits", type=int, default=100, help="Length of the random bit sequence for watermark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--mixed_precision", type=str, default="fp16", help="Mixed precision mode")
    parser.add_argument("--bce_loss_weight", type=float, default=1.0, help="Weight for the BCE loss")
    parser.add_argument("--l2_loss_weight", type=float, default=10.0, help="Weight for the L2 loss")
    parser.add_argument("--l2_loss_await", type=int, default=1000)
    parser.add_argument("--l2_loss_ramp", type=int, default=3000)
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps for the learning rate scheduler")

    #not sure
    parser.add_argument("--save_image_interval", type=int, default=10, help="Interval to save images during training")
    
    parser.add_argument('--save_root_dir', type=str, default='/scratch3/users/yufeng/Myproj/results', help="Root directory to save images")
    parser.add_argument('--log_type', type=str, default='wandb', help="Logging type")
    
    #parser.add_argument('--watermark_rate', type=float, default=0.1, help="Watermark rate")

    parser.add_argument('--debug_mode', type=bool, default=True, help="Whether to use debug mode")
    return parser.parse_args()

def generate_bitstring_watermark(bs, bit_length):
    msg = torch.randint(0, 2, (bs, bit_length)).float()
    return msg

def load_stegastamp_encoder(args):
    state_dict = torch.load(args.encoder_path)
    fingerpint_size = state_dict["secret_dense.weight"].shape[-1]

    HideNet = StegaStampEncoder(
        128,
        3,
        fingerprint_size=fingerpint_size,
        return_residual=False,
    )

    HideNet.load_state_dict(state_dict)

    return HideNet, fingerpint_size


def evaluate_stegastamp(args, dataset, accelerator, encoder, save_dir, epoch, global_step):
    device = accelerator.device
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    batch = next(iter(dataloader))
    images= batch["original_pixel_values"]

    images = images.to(device)
    msg = generate_bitstring_watermark(images.size(0), args.watermark_bits).to(device)

    with torch.no_grad():
        wm_images = encoder(msg, images)
    
    grid_original = torchvision.utils.make_grid(images, nrow=args.batch_size//2, normalize=True, scale_each=True)
    grid_clean_path = os.path.join(save_dir, "clean_images")
    os.makedirs(grid_clean_path, exist_ok=True)
    grid_clean_path = os.path.join(grid_clean_path, f"clean_images_epoch_{epoch+1}.png")
    torchvision.utils.save_image(grid_original, grid_clean_path)

    grid_watermarked = torchvision.utils.make_grid(wm_images, nrow=args.batch_size//2, normalize=True, scale_each=True)
    grid_watermarked_path = os.path.join(save_dir, "watermarked_images")
    os.makedirs(grid_watermarked_path, exist_ok=True)
    grid_watermarked_path = os.path.join(grid_watermarked_path, f"watermarked_images_epoch_{epoch+1}.png")
    torchvision.utils.save_image(grid_watermarked, grid_watermarked_path)



    # calculate clean generation PSNR and SSIM
    # transform tensor to NumPy and transform channel -> [B, H, W, C]
    images_np = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    wm_image_np = wm_images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    # 假設圖片已 normalize 到 [0,1]
    images_np = np.clip(images_np, 0, 1)
    wm_image_np = np.clip(wm_image_np, 0, 1)

    psnr_list = []
    ssim_list = []
    for i in range(images_np.shape[0]):
        psnr_value = peak_signal_noise_ratio(images_np[i], wm_image_np[i], data_range=1)
        ssim_value = structural_similarity(images_np[i], wm_image_np[i], channel_axis=-1, data_range=1)
        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    
    # report to wandb
    if accelerator.is_main_process:
        print(f"[Epoch {epoch+1}] PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
        wandb.log({
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "image": wandb.Image(grid_original, caption=f"Epoch {epoch+1}"),
            "wm_image": wandb.Image(grid_watermarked, caption=f"Epoch {epoch+1} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
        }, step=global_step)


def train_stegastamp(args, accelerator, save_dir):

    #dataset = LAIONDataset("tempertrash/laion_400m", resolution=args.image_resolution)
    dataset = InstructPix2PixDataset(args.dataset_name, resolution=args.image_resolution)

    encoder = StegaStampEncoder(
        args.image_resolution,
        3,
        args.watermark_bits,
        return_residual=False,
    )
    decoder = StegaStampDecoder(
        args.image_resolution,
        3,
        args.watermark_bits,
    )

    encoder, decoder = accelerator.prepare(encoder, decoder)
    encoder, decoder = encoder.to(accelerator.device), decoder.to(accelerator.device)

    optimizer = optim.AdamW(
        params=list(decoder.parameters()) + list(encoder.parameters()), lr=args.learning_rate
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.epochs * ((len(dataset) // args.batch_size) + 1),
    )
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    global_step = 0
    steps_since_l2_loss_activated = -1

    mse_loss_fn = nn.MSELoss()
    bce_loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(args.epochs):
        total_loss = 0.0
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=16
        )
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs}")

        for _, batch in enumerate(dataloader):
            global_step += 1
            images = batch["original_pixel_values"]

            bs = images.size(0)
            msg = generate_bitstring_watermark(bs, args.watermark_bits)
            
            clean_images, msg = images.to(accelerator.device), msg.to(accelerator.device)

            wm_images = encoder(msg, clean_images)

            residual = wm_images - clean_images          

            decoder_output = decoder(wm_images)

            l2_loss = mse_loss_fn(wm_images, clean_images)
            bce_loss = bce_loss_fn(decoder_output, msg) # reshape(-1)?

            l2_loss_weight = min(
                max(
                    0,
                    args.l2_loss_weight
                    * (steps_since_l2_loss_activated - args.l2_loss_await)
                    / args.l2_loss_ramp,
                ),
                args.l2_loss_weight,
            )
            bce_loss_weight = args.bce_loss_weight

            loss = l2_loss_weight * l2_loss + bce_loss_weight * bce_loss

            optimizer.zero_grad()

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            wm_msg_predicted = (decoder_output > 0).float()
            bitwise_accuracy = 1.0 - torch.mean(
                torch.abs(msg - wm_msg_predicted)
            )
            if steps_since_l2_loss_activated == -1:
                if bitwise_accuracy.item() > 0.9:
                    print(f"L2 loss activated at Epoch {epoch+1}")
                    steps_since_l2_loss_activated = 0
            else:
                steps_since_l2_loss_activated += 1
        
            progress_bar.update(1)
            total_loss += loss.item()
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        if (epoch + 1) % args.save_image_interval == 0:
            evaluate_stegastamp(args, dataset, accelerator, encoder, save_dir, epoch, global_step)

    encoder, decoder = accelerator.unwrap_model(encoder), accelerator.unwrap_model(decoder)

    return encoder, decoder


def main(args):
    accelerator = Accelerator(log_with=args.log_type, mixed_precision=args.mixed_precision)
    
    if not args.debug_mode:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    else:
        timestamp = "debug"

    save_dir = os.path.join(args.save_root_dir, timestamp)
    cwd = os.getcwd()
    save_ckpt_dir = os.path.join(cwd, "ckpt")
    

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        print("Device:", accelerator.device)    
        print("seed:", args.seed)
        wandb.init(project=args.project, name=args.exp_name, config=vars(args))
        accelerator.init_trackers(args.project, config=vars(args))
    
        os.makedirs(save_ckpt_dir, exist_ok=True)
        print("Save checkpoint to:", save_ckpt_dir)
    
    encoder, decoder = train_stegastamp(args, accelerator, save_dir)
    torch.save(encoder.state_dict(), os.path.join(save_ckpt_dir, "encoder.pt"))
    torch.save(decoder.state_dict(), os.path.join(save_ckpt_dir, "decoder.pt"))


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
