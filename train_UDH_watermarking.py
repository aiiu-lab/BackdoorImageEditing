import os
import argparse
import random
import math
import numpy as np
from datetime import datetime
from io import BytesIO

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage, ToTensor

from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

import wandb

from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# === 假設你已經有這兩個 model 檔案 (或類似) ===
from models.HidingUNet import UnetGenerator   # Hiding Network
from models.RevealNet import RevealNet        # Reveal Network

# === 你可使用與 StegaStamp 類似的 Dataset: InstructPix2PixDataset / ImageFolderDataset 等 ===
# 這裡給個示範: 簡易 ImageFolder + transforms
from torchvision.datasets import ImageFolder

def parse_args():
    parser = argparse.ArgumentParser(description="Train UDH (Universal Deep Hiding) for invisible watermark/steganography.")
    # ============= 基本參數 =============
    parser.add_argument("--project", type=str, default="UDH_Project", help="wandb project name")
    parser.add_argument("--exp_name", type=str, default="UDH_experiment", help="wandb experiment name")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no","fp16","bf16"])

    # ============= 資料 & 訓練超參數 =============
    parser.add_argument("--train_data_dir", type=str, default="./data/train", help="Path to training dataset folder")
    parser.add_argument("--val_data_dir", type=str, default="./data/val", help="Path to validation dataset folder")
    parser.add_argument("--image_resolution", type=int, default=128, help="Training image resolution")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.75, help="Weight for Reveal loss in total loss. total_loss = H_loss + beta*R_loss")

    parser.add_argument("--save_image_interval", type=int, default=5, help="Interval (epoch) to evaluate & save images")
    parser.add_argument("--save_root_dir", type=str, default="./results", help="Where to save outputs/checkpoints")
    parser.add_argument("--log_type", type=str, default="wandb", help="accelerator log type")

    args = parser.parse_args()
    return args

def create_dataloaders(args):
    """
    這裡示範使用 torchvision.datasets.ImageFolder, 
    你可自行改成自定義 dataset, 只要最後回傳 (cover, secret) 也可以。
    這裡我們簡單地將 train_data_dir 內的圖檔視為封面 or secret。
    實務上 UDH 可能需要兩份資料集 (cover, secret)，
    但這裡就示範: 從同一資料集中擷取兩批圖像來組合。
    """

    transform = transforms.Compose([
        transforms.Resize((args.image_resolution, args.image_resolution)),
        transforms.ToTensor(),
        # 轉到 [-1,1]
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    train_dataset = ImageFolder(args.train_data_dir, transform=transform)
    val_dataset   = ImageFolder(args.val_data_dir,   transform=transform)

    # 你也可以把 dataset 包一層, 讓 __getitem__ 同時 return 不同的 cover / secret
    # 這裡僅作簡單示範: 從同一 batch 分別抽前半做 secret, 後半做 cover

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size*2, shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(val_dataset,   batch_size=args.batch_size*2, shuffle=False, num_workers=4)
    return train_dataloader, val_dataloader

def forward_udh(Hnet, Rnet, cover, secret, criterion):
    """
    核心 forward pass:
    1) HidingNet(Hnet) 輸入 secret, 輸出隱藏訊號 S_e
    2) container = cover + S_e (pixel-wise 加法)
    3) RevealNet(Rnet) 輸入 container, 輸出解碼後 secret_hat
    4) 分別計算:
       - errH = MSE(container, cover)
       - errR = MSE(secret_hat, secret)
    """
    # 產生隱藏訊號 (S_e)
    s_e = Hnet(secret)

    # 容器圖 = cover + s_e
    container = cover + s_e

    # 解碼
    revealed_secret = Rnet(container)

    # 兩個 loss
    errH = criterion(container, cover)
    errR = criterion(revealed_secret, secret)
    return container, revealed_secret, errH, errR

def evaluate_udh(args, accelerator, Hnet, Rnet, val_dataloader, epoch, global_step, save_dir):
    """
    簡易評估:
    1. 拿 val batch, split 成 cover / secret
    2. forward pass
    3. 計算 PSNR, SSIM
    4. 存圖
    """
    Hnet.eval()
    Rnet.eval()

    batch = next(iter(val_dataloader))
    # batch size = 2 * (args.batch_size*2)? 取最前 batch 就好
    images, _ = batch
    images = images.to(accelerator.device)

    # 拆成 cover / secret
    half = images.size(0)//2
    cover  = images[:half]
    secret = images[half:]

    with torch.no_grad():
        container, revealed_secret, errH, errR = forward_udh(Hnet, Rnet, cover, secret, nn.MSELoss())

    # 轉回 [0,1] 方便計算PSNR,SSIM
    def to_01(tensor):
        # 原本 [-1,1]
        return (tensor.clamp(-1,1) + 1)/2

    cover_01     = to_01(cover).detach().cpu().numpy()
    secret_01    = to_01(secret).detach().cpu().numpy()
    container_01 = to_01(container).detach().cpu().numpy()
    revealed_01  = to_01(revealed_secret).detach().cpu().numpy()

    # shape: (B,C,H,W) -> PSNR, SSIM 需 channel_last
    cover_01     = np.transpose(cover_01, (0,2,3,1))
    secret_01    = np.transpose(secret_01, (0,2,3,1))
    container_01 = np.transpose(container_01, (0,2,3,1))
    revealed_01  = np.transpose(revealed_01, (0,2,3,1))

    # 計算PSNR, SSIM
    psnr_c_list = []
    ssim_c_list = []
    psnr_s_list = []
    ssim_s_list = []

    from math import log10
    for i in range(cover_01.shape[0]):
        c_psnr = peak_signal_noise_ratio(cover_01[i], container_01[i], data_range=1)
        c_ssim = structural_similarity(cover_01[i], container_01[i], channel_axis=-1, data_range=1)
        s_psnr = peak_signal_noise_ratio(secret_01[i], revealed_01[i], data_range=1)
        s_ssim = structural_similarity(secret_01[i], revealed_01[i], channel_axis=-1, data_range=1)
        psnr_c_list.append(c_psnr)
        ssim_c_list.append(c_ssim)
        psnr_s_list.append(s_psnr)
        ssim_s_list.append(s_ssim)

    avg_psnr_cover   = np.mean(psnr_c_list)
    avg_ssim_cover   = np.mean(ssim_c_list)
    avg_psnr_secret  = np.mean(psnr_s_list)
    avg_ssim_secret  = np.mean(ssim_s_list)

    if accelerator.is_main_process:
        print(f"[Val Epoch {epoch+1}]  H_loss={errH.item():.4f}, R_loss={errR.item():.4f}")
        print(f"   Cover:  PSNR={avg_psnr_cover:.2f}, SSIM={avg_ssim_cover:.4f}")
        print(f"   Secret: PSNR={avg_psnr_secret:.2f}, SSIM={avg_ssim_secret:.4f}")

        # log to wandb
        logs = {
            "val/H_loss": errH.item(),
            "val/R_loss": errR.item(),
            "val/cover_psnr": avg_psnr_cover,
            "val/cover_ssim": avg_ssim_cover,
            "val/secret_psnr": avg_psnr_secret,
            "val/secret_ssim": avg_ssim_secret
        }
        wandb.log(logs, step=global_step)

        # 存圖
        # 將 cover, container, residual, secret, revealed_secret 做成grid
        def to_tensor_01(ndarr):
            # shape (B,H,W,C)
            ndarr = np.transpose(ndarr, (0,3,1,2))
            return torch.from_numpy(ndarr)

        # 取 batch 前 4 張可視化
        show_num = min(4, cover.size(0))
        cover_t     = to_tensor_01(cover_01[:show_num])
        container_t = to_tensor_01(container_01[:show_num])
        secret_t    = to_tensor_01(secret_01[:show_num])
        revealed_t  = to_tensor_01(revealed_01[:show_num])

        residual_cover = (container_t - cover_t).abs() * 5.0  # 乘個係數方便看
        residual_secret = (revealed_t - secret_t).abs() * 5.0

        # [cover, container, residual_cover, secret, revealed, residual_secret]
        grid = torch.cat([cover_t, container_t, residual_cover, secret_t, revealed_t, residual_secret], dim=0)
        grid_path = os.path.join(save_dir, f"val_epoch_{epoch+1}_grid.png")
        torchvision.utils.save_image(grid, grid_path, nrow=show_num, padding=2)
        print(f"Val images saved to {grid_path}")

    Hnet.train()
    Rnet.train()

def train_udh(args, accelerator, save_dir):
    """
    主要訓練函式:
    1. 構建 Hnet, Rnet
    2. MSELoss
    3. 讀取 dataloader, split batch => cover, secret
    4. forward => container => reveal => loss => backward
    5. 每隔 N epoch evaluate
    """

    # model
    Hnet = UnetGenerator(input_nc=3, output_nc=3, num_downs=5, norm_layer=nn.BatchNorm2d, output_function=nn.Tanh)
    Rnet = RevealNet(input_nc=3, output_nc=3, nhf=64, norm_layer=nn.BatchNorm2d, output_function=nn.Sigmoid) 

    
    Hnet, Rnet = accelerator.prepare(Hnet, Rnet)
    Hnet.train()
    Rnet.train()

    # 建立 optimizer
    optimizer = optim.AdamW(list(Hnet.parameters()) + list(Rnet.parameters()), lr=args.learning_rate)
    # 也可加 scheduler, 例如 torch.optim.lr_scheduler.StepLR, etc.
    # 這裡略

    criterion = nn.MSELoss()

    # 取得 dataloader
    train_dataloader, val_dataloader = create_dataloaders(args)

    global_step = 0
    for epoch in range(args.epochs):
        progress_bar = tqdm(train_dataloader, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs}")

        for i, (images, _) in enumerate(progress_bar):
            # images shape: (B*2, 3, H, W)
            # 其中 B= args.batch_size
            # 拆成 cover / secret
            # 這裡假設 B*2 一半做 cover, 一半做 secret
            # 也可隨機 shuffle, 這裡簡化
            half = images.size(0)//2
            cover  = images[:half].to(accelerator.device)
            secret = images[half:].to(accelerator.device)

            container, revealed_secret, errH, errR = forward_udh(Hnet, Rnet, cover, secret, criterion)
            loss = errH + args.beta * errR

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            logs = {
                "train/loss": loss.item(),
                "train/H_loss": errH.item(),
                "train/R_loss": errR.item(),
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            global_step += 1

        # evaluate & save images
        if (epoch+1) % args.save_image_interval == 0:
            evaluate_udh(args, accelerator, Hnet, Rnet, val_dataloader, epoch, global_step, save_dir)

    # unwrap
    Hnet = accelerator.unwrap_model(Hnet)
    Rnet = accelerator.unwrap_model(Rnet)
    return Hnet, Rnet

def main(args):
    
    accelerator = Accelerator(log_with=args.log_type, mixed_precision=args.mixed_precision)
    if accelerator.is_main_process:
        wandb.init(project=args.project, name=args.exp_name, config=vars(args))
        accelerator.init_trackers(args.project, config=vars(args))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    save_dir = os.path.join(args.save_root_dir, f"udh_{timestamp}")
    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)

    # training
    Hnet, Rnet = train_udh(args, accelerator, save_dir)

    # save checkpoint
    if accelerator.is_main_process:
        torch.save(Hnet.state_dict(), os.path.join(save_dir, "Hnet.pth"))
        torch.save(Rnet.state_dict(), os.path.join(save_dir, "Rnet.pth"))
        print("UDH model saved!")

if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
