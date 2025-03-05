import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import lpips  # pip install lpips
from diffusers import StableDiffusionImg2ImgPipeline, UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator
from tqdm.auto import tqdm
import argparse
import wandb
import numpy as np
from datetime import datetime
import random
from PIL import Image

from models.Message_model import MessageModel
from models.StegaStamp import StegaStampEncoder, StegaStampDecoder
from util import set_seed
from dataset import get_celeba_hq_dataset, ClelebAHQWatermarkedDataset, CelebADataset
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional diffusion model with spatial watermark protection")
    parser.add_argument("--project", type=str, default="Watermark_Baddiffusion", help="Project name for wandb")
    parser.add_argument("--dataset", type=str, default="celeba", choices=["celeba", "celeba-hq"], help="Dataset to use")
    parser.add_argument("--data_path", type=str, default="/scratch3/users/yufeng/Myproj/datasets/celeba", help="Path to the dataset")
    parser.add_argument("--encoder_path", type=str, default="/scratch3/users/yufeng/Myproj/ckpt/CelebA_128x128_encoder.pth", help="Path to the encoder")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for the optimizer")
    parser.add_argument("--watermark_bits", type=int, default=64, help="Length of the random bit sequence for watermark")
    parser.add_argument("--alpha_lpips", type=float, default=1.0, help="Weight for the clean LPIPS loss")
    parser.add_argument("--alpha_w_lpips", type=float, default=1.0, help="Weight for backdoor LPIPS loss")
    parser.add_argument("--loss_type", type=str, default="mse_noise", choices=["mse_noise", "mse_lpips_clean","mse_lpips_backdoor"], help="Loss function type")
    
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps for training and sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--mixed_precision", type=str, default="fp16", help="Mixed precision mode")
    
    parser.add_argument("--save_image_interval", type=int, default=10, help="Interval to save images during training")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps for the learning rate scheduler")
    
    parser.add_argument('--save_watermarked_imgs', type=int, default=1, help="Whether to save watermarked images")
    parser.add_argument('--save_root_dir', type=str, default='/scratch3/users/yufeng/Myproj/results', help="Root directory to save images")
    parser.add_argument('--log_type', type=str, default='wandb', help="Logging type")
    
    parser.add_argument('--watermark_rate', type=float, default=0.1, help="Watermark rate")
    
    parser.add_argument('--only_sde_inference', action='store_true', help="Only Use SDEdit for inference")
    parser.add_argument('--sde_strength', type=float, default=0.5, help="Noise injection strength for SDEdit (0~1)")
    
    parser.add_argument('--debug_mode', type=bool, default=False, help="Whether to use debug mode")
    parser.add_argument("--phase", type=str, default="unet", choices=["watermark", "unet", "joint"], help="Training phase to run")
    return parser.parse_args()


def evaluate_settings(accelerator, dataloader, pipeline, message_model, latent_scaling, msg, save_dir, epoch):
    """
    驗證 clean setting 與加 watermark 的效果：
      - 將 test_image 轉成 latent，
      - 對 clean setting，不添加 watermark，
      - 對 watermark setting，將 latent 加上 watermark，
      - 接著通過 diffusion 模型（採樣流程）得到最終 latent，
      - 再用 VAE decode 成圖像，
      - 最後將兩種結果存成圖片。
    """
    device = accelerator.device

    batch = next(iter(dataloader))
    images = batch["image"].to(device)
    
    # Step 1. 生成 encoder_hidden_states：採用空文本作為條件 (unconditional)
    text_input = [""] * images.size(0)

    # clean image regeneration
    with torch.no_grad():
        generated_clean_images = pipeline(prompt=text_input, image=images, strength=0.5, guidance_scale=1, output_type="pt").images


    # watermark setting
    

    # 將原圖與重構圖視覺化
    grid_orig = torchvision.utils.make_grid(generated_clean_images, nrow=4, normalize=True, scale_each=True)
    
    grid_path = os.path.join(save_dir, f"recon_epoch_{epoch+1}.png")
    torchvision.utils.save_image(grid_orig, grid_path)

    # 計算 PSNR 與 SSIM
    # 轉換張量到 NumPy 陣列並轉置維度為 [B, H, W, C]
    images_np = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    recon_np = generated_clean_images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    # 假設圖片已 normalize 到 [0,1]
    images_np = np.clip(images_np, 0, 1)
    recon_np = np.clip(recon_np, 0, 1)

    psnr_list = []
    ssim_list = []
    for i in range(images_np.shape[0]):
        psnr_value = peak_signal_noise_ratio(images_np[i], recon_np[i], data_range=1)
        ssim_value = structural_similarity(images_np[i], recon_np[i], channel_axis=-1, data_range=1)
        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)

    print(f"[Epoch {epoch+1}] PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")
    
    # 記錄至 wandb
    if accelerator.is_main_process:
        wandb.log({
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "reconstruction": wandb.Image(grid_orig, caption=f"Epoch {epoch+1} | PSNR: {avg_psnr:.2f} dB SSIM: {avg_ssim:.4f}")
        }, step=epoch)



def train_joint_model(args, accelerator, save_dir):
    
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline.to(accelerator.device)
    vae = pipeline.vae
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    diffusion_model = pipeline.unet
    diffusion_model.train()

    noise_sched = pipeline.scheduler

    text_input = ""
    text_inputs = pipeline.tokenizer(
        text_input,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(accelerator.device)

    encoder_hidden_states = pipeline.text_encoder(
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"]
    ).last_hidden_state.to(accelerator.device)

    encoder_hidden_states = encoder_hidden_states.repeat(args.batch_size, 1, 1)

    target_resolution = 256
    
    dataset = ClelebAHQWatermarkedDataset(watermark_rate=args.watermark_rate, resolution=target_resolution)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    message_model = MessageModel(args.watermark_bits, 4, target_resolution // 8, target_resolution // 8)
    message_model = accelerator.prepare(message_model)
    message_model = accelerator.unwrap_model(message_model)

    optimizer = optim.AdamW(list(diffusion_model.parameters()) + list(message_model.parameters()), lr=args.learning_rate)

    lr_sched = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader) * args.epochs),
    )
    optimizer, lr_sched = accelerator.prepare(optimizer, lr_sched)
    
    mse_loss_fn = nn.MSELoss()
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(accelerator.device)
    msg_loss_fn = nn.BCEWithLogitsLoss()
    latent_scaling = 0.18215

    msg = torch.randint(0, 2, (1,args.watermark_bits)).float().to(accelerator.device)
    msg = msg.repeat(args.batch_size, 1)
    for epoch in range(args.epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Watermark AE Epoch {epoch+1}/{args.epochs}")
        for _, batch in enumerate(dataloader):
            images, is_watermarked, targets = batch["image"], batch["is_watermarked"], batch["target"]
            images, is_watermarked, targets = images.to(accelerator.device), is_watermarked.to(accelerator.device), targets.to(accelerator.device)
            batch_size = images.size(0)
            optimizer.zero_grad()

            with torch.no_grad():
                latent_dist = vae.encode(images).latent_dist
                latents = latent_dist.sample() * latent_scaling

            
            watermark = message_model.encode(msg)
            
            watermarked_latents = latents.clone()
            watermarked_latents[is_watermarked] = latents[is_watermarked] + watermark[is_watermarked]#args.alpha *
            
            # 取整數 timestep
            t = torch.randint(0, noise_sched.config.num_train_timesteps, (batch_size,), device=accelerator.device).long()
            noise = torch.randn_like(watermarked_latents)
            noised_latents = noise_sched.add_noise(watermarked_latents, noise, t)
            
            # (e) Diffusion Model 預測乾淨 latent（假設模型直接預測 x0）
            model_output = diffusion_model(noised_latents, t, encoder_hidden_states)
            pred_latents = model_output["sample"]

            
            recon = vae.decode(pred_latents / latent_scaling).sample
            msg_hat = message_model.decode(pred_latents)
            
            loss_msg = msg_loss_fn(msg_hat, msg)
            loss_lpips = lpips_loss_fn(recon, targets).mean()
            #loss_mse = mse_loss_fn(pred_latents, watermarked_latents)
            
            lambda_lpips = 1.0
            loss = loss_msg + lambda_lpips * loss_lpips #+ loss_mse

            accelerator.backward(loss, retain_graph=True)
            optimizer.step()
            lr_sched.step()

            progress_bar.update(1)
            total_loss += loss.item()
            logs = {
                "loss_msg": loss_msg.detach().item(),
                "loss_lpips": loss_lpips.detach().item(),
                #"loss_mse": loss_mse.item(),
                "lr": lr_sched.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=epoch)

        print(f"Watermark AE Epoch [{epoch+1}/{args.epochs}] Average Loss: {total_loss/len(dataloader):.4f}")
        #evaluate_watermark(args, accelerator, vae, message_model, latent_scaling, dataloader, save_dir, epoch, msg)
        evaluate_settings(accelerator, dataloader, pipeline, message_model, latent_scaling, msg, save_dir, epoch)

    return message_model, diffusion_model

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


def evaluate_stegastamp(args, pipeline, accelerator, stegastamp_encoder, msg, save_dir, epoch):
    device = accelerator.device

    if args.dataset == "celeba":
        dataset = CelebADataset(args.data_path, resolution=128, stage="test")
    elif args.dataset == "celeba-hq":
        dataset = ClelebAHQWatermarkedDataset(resolution=128)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    batch = next(iter(dataloader))
    images, targets = batch["image"].to(device), batch["target"].to(device)

    # Step 1. 生成 encoder_hidden_states：採用空文本作為條件 (unconditional)
    text_input = [""] * images.size(0)
    
    # clean image regeneration
    with torch.no_grad():
        generated_clean_images = pipeline(prompt=text_input, image=images, guidance_scale=1, output_type="pt").images
        

    # watermark setting
    with torch.no_grad():
        watermarked_images = stegastamp_encoder(msg, images)
        generated_backdoor_images = pipeline(prompt=text_input, image=watermarked_images, guidance_scale=1, output_type="pt").images
    
    # 將原圖與重構圖視覺化
    grid_recon = torchvision.utils.make_grid(generated_clean_images, nrow=6, normalize=True, scale_each=True)
    grid_backdoor = torchvision.utils.make_grid(generated_backdoor_images, nrow=6, normalize=True, scale_each=True)
    grid = torch.cat([grid_recon, grid_backdoor], dim=1)
    grid_path = os.path.join(save_dir, f"recon_backdoor_epoch_{epoch+1}.png")
    torchvision.utils.save_image(grid, grid_path)

    # calculate clean generation PSNR and SSIM
    # transform tensor to NumPy and transform channel -> [B, H, W, C]
    images_np = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    recon_np = generated_clean_images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    # 假設圖片已 normalize 到 [0,1]
    images_np = np.clip(images_np, 0, 1)
    recon_np = np.clip(recon_np, 0, 1)

    psnr_list = []
    ssim_list = []
    for i in range(images_np.shape[0]):
        psnr_value = peak_signal_noise_ratio(images_np[i], recon_np[i], data_range=1)
        ssim_value = structural_similarity(images_np[i], recon_np[i], channel_axis=-1, data_range=1)
        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)
    clean_avg_psnr = np.mean(psnr_list)
    clean_avg_ssim = np.mean(ssim_list)
    
    
    targets_np = targets.detach().cpu().numpy().transpose(0, 2, 3, 1)
    targets_recon_np = generated_backdoor_images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    # 假設圖片已 normalize 到 [0,1]
    targets_np = np.clip(targets_np, 0, 1)
    targets_recon_np = np.clip(targets_recon_np, 0, 1)

    psnr_list = []
    ssim_list = []
    for i in range(targets_np.shape[0]):
        psnr_value = peak_signal_noise_ratio(targets_np[i], targets_recon_np[i], data_range=1)
        ssim_value = structural_similarity(targets_np[i], targets_recon_np[i], channel_axis=-1, data_range=1)
        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)
    target_avg_psnr = np.mean(psnr_list)
    target_avg_ssim = np.mean(ssim_list)

    
    
    # 記錄至 wandb
    if accelerator.is_main_process:
        print(f"[Epoch {epoch+1}] clean PSNR: {clean_avg_psnr:.2f} dB, clean SSIM: {clean_avg_ssim:.4f}")
        print(f"[Epoch {epoch+1}] target PSNR: {target_avg_psnr:.2f} dB, target SSIM: {target_avg_ssim:.4f}")
        wandb.log({
            "clean_psnr": clean_avg_psnr,
            "clean_ssim": clean_avg_ssim,
            "target_psnr": target_avg_psnr,
            "target_ssim": target_avg_ssim,
            "reconstruction": wandb.Image(grid, caption=f"Epoch {epoch+1} | clean PSNR: {clean_avg_psnr:.2f} dB | clean SSIM: {clean_avg_ssim:.4f} | target PSNR: {target_avg_psnr:.2f} dB | target SSIM: {target_avg_ssim:.4f}")
        }, step=epoch)

def train_unet_model(args, accelerator, save_dir):

    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline.to(accelerator.device)
    vae = pipeline.vae
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    diffusion_model = pipeline.unet
    diffusion_model.train()

    
    text_input = ""
    text_inputs = pipeline.tokenizer(
        text_input,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    ).to(accelerator.device)

    one_text_encoder_hidden_states = pipeline.text_encoder(
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"]
    ).last_hidden_state.to(accelerator.device)

    target_resolution = 128
    
    if args.dataset == "celeba":
        dataset = CelebADataset(args.data_path, target_resolution, "train")
    elif args.dataset == "celeba-hq":
        dataset = ClelebAHQWatermarkedDataset(resolution=target_resolution)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    stegastamp_encoder, fingerpint_size = load_stegastamp_encoder(args)
    stegastamp_encoder = accelerator.prepare(stegastamp_encoder)
    stegastamp_encoder = accelerator.unwrap_model(stegastamp_encoder)
    stegastamp_encoder = stegastamp_encoder.to(accelerator.device)

    optimizer = optim.AdamW(diffusion_model.parameters(), lr=args.learning_rate)

    lr_sched = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader) * args.epochs),
    )
    optimizer, lr_sched = accelerator.prepare(optimizer, lr_sched)
    
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(accelerator.device)
    latent_scaling = 0.18215

    msg = torch.randint(0, 2, (1,fingerpint_size)).float().to(accelerator.device)
    msg = msg.repeat(args.batch_size, 1)
    global_step = 0
    for epoch in range(args.epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}/{args.epochs}")
        noise_sched = DDPMScheduler.from_config(pipeline.scheduler.config)

        for _, batch in enumerate(dataloader):
            images, targets = batch["image"], batch["target"]
            images, targets = images.to(accelerator.device), targets.to(accelerator.device)
            
            bs = images.size(0)

            with torch.no_grad():
                latent_dist = vae.encode(images).latent_dist
                latents = latent_dist.sample() * latent_scaling

            t = torch.randint(0, noise_sched.config.num_train_timesteps, (bs,), device=accelerator.device).long()
            noise = torch.randn_like(latents).to(accelerator.device)
            
            encoder_hidden_states = one_text_encoder_hidden_states.repeat(bs, 1, 1)
            
            with accelerator.accumulate(diffusion_model):
                noisy_latents = noise_sched.add_noise(latents, noise, t)
                # if error print error information
                try:
                    noise_pred = diffusion_model(noisy_latents, t, encoder_hidden_states=encoder_hidden_states,return_dict=False)[0]
                except:
                    print("noisy_latents shape:", noisy_latents.shape)
                    print("t shape:", t.shape)
                    print("encoder_hidden_states shape:", encoder_hidden_states.shape)
                    print("noise shape:", noise.shape)
                    raise ValueError("Error in forward pass of diffusion model")
                # 噪聲預測損失 (例如 MSE Loss)
                clean_loss_noise = F.mse_loss(noise_pred, noise)
                
                # ===== Step 3: 影像重建損失 =====
                # 利用 scheduler 進行一步去噪以獲得預測的 latent 表示
                if args.loss_type.startswith("mse_lpips"):
                    try:
                        pred_latents = torch.stack([
                            noise_sched.step(noise_pred[i], t[i], noisy_latents[i], return_dict=True).prev_sample 
                            for i in range(bs)
                        ])
                    except:
                        print("noisy_latents shape:", noisy_latents.shape)
                        print("t shape:", t.shape)
                        print("noise_pred shape:", noise_pred.shape)
                        print("noise shape:", noise.shape)
                        raise ValueError("Error in step of noise scheduler")
                    recon_images = vae.decode(pred_latents / latent_scaling).sample
                    clean_image_loss_lpips = lpips_loss_fn(recon_images, images).mean()
                
                
                #watermarked training pipeline
                if args.loss_type == "mse_lpips_backdoor":
                    with torch.no_grad():
                        watermarked_images = stegastamp_encoder(msg, images)
                        watermarked_latent_dist = vae.encode(watermarked_images).latent_dist
                        watermarked_latents = watermarked_latent_dist.sample() * latent_scaling

                    watermarked_noisy_latents = noise_sched.add_noise(watermarked_latents, noise, t)
                    
                    watermarked_noise_predict = diffusion_model(watermarked_noisy_latents, t, encoder_hidden_states)["sample"]
                    
                    #watermarked_loss_noise = F.mse_loss(watermarked_model_output, noise)

                    watermarked_pred_latents = torch.stack([
                        noise_sched.step(watermarked_noise_predict[i], t[i], watermarked_noisy_latents[i], return_dict=True).prev_sample
                        for i in range(bs)
                    ])
                    watermarked_recon = vae.decode(watermarked_pred_latents / latent_scaling).sample
                    watermarked_loss_lpips = lpips_loss_fn(watermarked_recon, targets).mean()
            
            
                alpha_lpips, alpha_w_lpips = args.alpha_lpips, args.alpha_w_lpips
                # if args.backdoor_watermark_predict:
                #     loss = alpha_lpips * clean_image_loss_lpips + alpha_w_lpips * watermarked_loss_lpips + clean_loss_noise 
                # else:
                #     loss = alpha_lpips * clean_image_loss_lpips + clean_loss_noise
                if args.loss_type == "mse_noise":
                    loss = clean_loss_noise
                elif args.loss_type == "mse_lpips_clean":
                    loss = clean_loss_noise + clean_image_loss_lpips
                elif args.loss_type == "mse_lpips_backdoor":
                    loss = clean_loss_noise + alpha_lpips * clean_image_loss_lpips + alpha_w_lpips * watermarked_loss_lpips

            
                accelerator.backward(loss, retain_graph=True) # retain_graph = True
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            total_loss += loss.item()
            logs = {
                "loss": loss.detach().item(),
                # "w_lpips": watermarked_loss_lpips.detach().item(),
                #"loss_mse": loss_mse.item(),
                "lr": lr_sched.get_last_lr()[0],
                "step": global_step
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            print(f"Epoch [{epoch+1}/{args.epochs}] Average Loss: {total_loss/len(dataloader):.4f}")
            pipeline = StableDiffusionImg2ImgPipeline(unet=accelerator.unwrap_model(diffusion_model), scheduler=noise_sched, vae=vae, tokenizer=pipeline.tokenizer, text_encoder=pipeline.text_encoder, feature_extractor=pipeline.feature_extractor, safety_checker=pipeline.safety_checker, requires_safety_checker=False)
            #evaluate_watermark(args, accelerator, vae, message_model, latent_scaling, dataloader, save_dir, epoch, msg)
            evaluate_stegastamp(args, pipeline, accelerator, stegastamp_encoder, msg, save_dir, epoch)

    return diffusion_model


def evaluate_pretrain(args, accelerator, save_dir):
    device = accelerator.device

    # evaluate
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipeline.to(accelerator.device)

    stegastamp_encoder, fingerpint_size = load_stegastamp_encoder(args)
    stegastamp_encoder = accelerator.prepare(stegastamp_encoder)
    stegastamp_encoder = accelerator.unwrap_model(stegastamp_encoder)
    stegastamp_encoder = stegastamp_encoder.to(accelerator.device)

    msg = torch.randint(0, 2, (1,fingerpint_size)).float().to(accelerator.device)
    msg = msg.repeat(args.batch_size, 1)

    #dataset = CelebADataset(args.data_path, resolution=128, stage="test")
    dataset = ClelebAHQWatermarkedDataset(resolution=128)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    batch = next(iter(dataloader))
    images, targets = batch["image"].to(device), batch["target"].to(device)

    # Step 1. 生成 encoder_hidden_states：採用空文本作為條件 (unconditional)
    text_input = ["Turn the person into angel"] * images.size(0)
    
    # clean image regeneration
    with torch.no_grad():
        generated_clean_images = pipeline(prompt=text_input, image=images, guidance_scale=7.5, output_type="pt").images
        

    # watermark setting
    with torch.no_grad():
        watermarked_images = stegastamp_encoder(msg, images)
        generated_backdoor_images = pipeline(prompt=text_input, image=watermarked_images, guidance_scale=7.5, output_type="pt").images
    
    # 將原圖與重構圖視覺化
    grid_recon = torchvision.utils.make_grid(generated_clean_images, nrow=6, normalize=True, scale_each=True)
    grid_backdoor = torchvision.utils.make_grid(generated_backdoor_images, nrow=6, normalize=True, scale_each=True)
    grid = torch.cat([grid_recon, grid_backdoor], dim=1)
    grid_path = os.path.join(save_dir, f"pretrain_recon_backdoor.png")
    torchvision.utils.save_image(grid, grid_path)

    # calculate clean generation PSNR and SSIM
    # transform tensor to NumPy and transform channel -> [B, H, W, C]
    images_np = images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    recon_np = generated_clean_images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    # 假設圖片已 normalize 到 [0,1]
    images_np = np.clip(images_np, 0, 1)
    recon_np = np.clip(recon_np, 0, 1)

    psnr_list = []
    ssim_list = []
    for i in range(images_np.shape[0]):
        psnr_value = peak_signal_noise_ratio(images_np[i], recon_np[i], data_range=1)
        ssim_value = structural_similarity(images_np[i], recon_np[i], channel_axis=-1, data_range=1)
        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)
    clean_avg_psnr = np.mean(psnr_list)
    clean_avg_ssim = np.mean(ssim_list)
    
    
    targets_np = targets.detach().cpu().numpy().transpose(0, 2, 3, 1)
    targets_recon_np = generated_backdoor_images.detach().cpu().numpy().transpose(0, 2, 3, 1)
    # 假設圖片已 normalize 到 [0,1]
    targets_np = np.clip(targets_np, 0, 1)
    targets_recon_np = np.clip(targets_recon_np, 0, 1)

    psnr_list = []
    ssim_list = []
    for i in range(targets_np.shape[0]):
        psnr_value = peak_signal_noise_ratio(targets_np[i], targets_recon_np[i], data_range=1)
        ssim_value = structural_similarity(targets_np[i], targets_recon_np[i], channel_axis=-1, data_range=1)
        psnr_list.append(psnr_value)
        ssim_list.append(ssim_value)
    target_avg_psnr = np.mean(psnr_list)
    target_avg_ssim = np.mean(ssim_list)

    print(f"clean PSNR: {clean_avg_psnr:.2f} dB, clean SSIM: {clean_avg_ssim:.4f}")
    print(f"target PSNR: {target_avg_psnr:.2f} dB, target SSIM: {target_avg_ssim:.4f}")


def main(args):
    accelerator = Accelerator(log_with=args.log_type, mixed_precision=args.mixed_precision)
    
    if not args.debug_mode:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    else:
        timestamp = "debug"
    save_dir = os.path.join(args.save_root_dir, timestamp)
    # os.makedirs(save_dir, exist_ok=True)

    # evaluate_pretrain(args, accelerator, save_dir)
    # return

    if accelerator.is_main_process:
        os.makedirs(save_dir, exist_ok=True)
        print("Device:", accelerator.device)    
        print("phase:", args.phase)
        print("seed:", args.seed)
        wandb.init(project=args.project, name="WatermarkModel", config=vars(args))
        accelerator.init_trackers(args.project, config=vars(args))

    if args.phase == "unet":
        unet = train_unet_model(args, accelerator, save_dir)
        torch.save(unet.state_dict(), os.path.join(save_dir, "unet.pt"))

    elif args.phase == "joint":
        message_model, diffusion_model = train_joint_model(args, accelerator, save_dir)
        
        torch.save(message_model.encoder.state_dict(), os.path.join(save_dir, "message_encoder.pt"))
        torch.save(message_model.decoder.state_dict(), os.path.join(save_dir, "message_decoder.pt"))
        torch.save(diffusion_model.state_dict(), os.path.join(save_dir, "diffusion_model.pt"))
    


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    main(args)
