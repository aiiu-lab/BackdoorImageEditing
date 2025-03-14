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
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
from datasets import load_dataset

#from models.Message_model import MessageModel
from models.StegaStamp import StegaStampEncoder, StegaStampDecoder
# from util import set_seed
from dataset import InstructPix2PixDataset # get_celeba_hq_dataset, ClelebAHQWatermarkedDataset, CelebADataset, LAIONDataset, 
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

distortion_strength_paras = dict(
    brightness=(1, 2),
    contrast=(1, 2),
    blurring=(0, 20),
    noise=(0, 0.1),
    compression=(10, 90),
)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional diffusion model with spatial watermark protection")
    #wandb
    parser.add_argument("--project", type=str, default="Watermark_Baddiffusion", help="Project name for wandb")
    parser.add_argument("--exp_name", type=str, default="train_wm_encoder_decoder", help="Experiment name for wandb")

    # data and model paths
    parser.add_argument("--dataset", type=str, default="celeba", choices=["celeba", "celeba-hq"], help="Dataset to use")
    parser.add_argument("--dataset_name", type=str, default="timbrooks/instructpix2pix-clip-filtered" , help="Dataset name")
    parser.add_argument("--max_train_samples", type=int, default=6000, help="Maximum number of training samples to use")
    parser.add_argument("--eval_samples", type=int, default=1000, help="Number of samples to use for evaluation")
    
    # training config
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=80, help="Max Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for the optimizer")
    parser.add_argument("--image_resolution", type=int, default=256, help="Resolution of the images")
    parser.add_argument("--watermark_bits", type=int, default=100, help="Length of the random bit sequence for watermark")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--mixed_precision", type=str, default="fp16", help="Mixed precision mode")
    parser.add_argument("--bce_loss_weight", type=float, default=1, help="Weight for the BCE loss")
    parser.add_argument("--lpips_init_loss_weight", type=float, default=0.0, help="Initial weight for the lpips loss")
    parser.add_argument("--l2_loss_weight", type=float, default=10.0, help="Weight for the LPIPS loss")
    parser.add_argument("--l2_loss_await", type=int, default=1000)
    parser.add_argument("--l2_loss_ramp", type=int, default=3000)
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps for the learning rate scheduler")
    parser.add_argument("--distortion_type", type=str, default="random", choices=["clean","random","brightness","contrast","blurring","noise","compression"], help="Types of distortions to apply")

    #not sure
    parser.add_argument("--save_image_interval", type=int, default=5, help="Interval to save images during training")
    
    parser.add_argument('--save_root_dir', type=str, default='/scratch3/users/yufeng/Myproj/results', help="Root directory to save images")
    parser.add_argument('--log_type', type=str, default='wandb', help="Logging type")
    
    #parser.add_argument('--watermark_rate', type=float, default=0.1, help="Watermark rate")

    parser.add_argument('--debug_mode', type=bool, default=True, help="Whether to use debug mode")
    return parser.parse_args()

def generate_bitstring_watermark(bs, bit_length):
    msg = torch.randint(0, 2, (bs, bit_length)).float()
    return msg

def pil_to_tensor(pil_img):
    return transforms.ToTensor()(pil_img)

def tensor_to_pil(tensor):
    # from tensor [-1, 1] transfer to [0,255] and modify dim
    
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor + 1) / 2 * 255
    tensor = np.clip(tensor, 0, 255).astype(np.uint8)
    # (3, H, W) -> (H, W, 3)
    tensor = np.transpose(tensor, (1, 2, 0))
    return Image.fromarray(tensor)

def get_datasets(args):
    
    full_dataset = InstructPix2PixDataset(args.dataset_name, args.image_resolution)
    
    train_dataset = full_dataset.select(range(args.max_train_samples))
    test_dataset = full_dataset.select(range(args.max_train_samples, args.max_train_samples + args.eval_samples))
    return train_dataset, test_dataset

def apply_single_distortion(image, distortion_type):

    if distortion_type == "brightness":
        factor = random.uniform(*distortion_strength_paras["brightness"])
        enhancer = ImageEnhance.Brightness(image)
        distorted_image = enhancer.enhance(factor)
    elif distortion_type == "contrast":
        factor = random.uniform(*distortion_strength_paras["contrast"])
        enhancer = ImageEnhance.Contrast(image)
        distorted_image = enhancer.enhance(factor)
    elif distortion_type == "blurring":
        # 對於模糊，強度轉換為 kernel size (例如 0 到 20)
        kernel_size = random.randint(*distortion_strength_paras["blurring"])
        distorted_image = image.filter(ImageFilter.GaussianBlur(kernel_size))
    elif distortion_type == "noise":
        std = random.uniform(*distortion_strength_paras["noise"])
        image_tensor = pil_to_tensor(image)
        noise = torch.randn(image_tensor.size()) * std
        noisy_tensor = (image_tensor + noise).clamp(0, 1)
        distorted_image = transforms.ToPILImage()(noisy_tensor)
    elif distortion_type == "compression":
        quality = random.randint(*distortion_strength_paras["compression"]) # randint
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=quality)
        buffered.seek(0)
        distorted_image = Image.open(buffered)
    else:
        raise ValueError(f"Unknown distortion type: {distortion_type}")
    return distorted_image

def image_distortion(images, distortion_type):
    if distortion_type == "clean":
        return (images + 1) / 2 # -> [0, 1]
    distorted_images = []
    for i in range(images.size(0)):
        if distortion_type == "random":
            distortion_type = random.choice(list(distortion_strength_paras.keys()))
        pil_img = tensor_to_pil(images[i])
        distorted_pil = apply_single_distortion(pil_img, distortion_type)
        distorted_tensor = pil_to_tensor(distorted_pil)
        distorted_images.append(distorted_tensor)
    distorted_images = torch.stack(distorted_images).to(images.device)
    return distorted_images

def evaluate_stegastamp(args, dataloader, accelerator, encoder, decoder, save_dir, epoch, global_step):
    device = accelerator.device
    
    # dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

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

    # --- Calculate bit accuracy for original watermarked images ---
    decoder_output_orig = decoder(wm_images)
    wm_msg_predicted_orig = (decoder_output_orig > 0).float()
    bit_acc_orig = 1.0 - torch.mean(torch.abs(msg - wm_msg_predicted_orig))
    
    # --- Calculate bit accuracy for each distortion ---
    bit_acc_dist = {}
    # Iterate over each distortion type defined in distortion_strength_paras
    for dist_type in distortion_strength_paras.keys():
        # Apply distortion to watermarked images
        distorted_wm = image_distortion(wm_images, distortion_type=dist_type)
        decoder_output_dist = decoder(distorted_wm)
        wm_msg_predicted_dist = (decoder_output_dist > 0).float()
        bit_acc = 1.0 - torch.mean(torch.abs(msg - wm_msg_predicted_dist))
        bit_acc_dist[dist_type] = bit_acc.item()



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
        print(f"[Epoch {epoch+1}] PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}, BitAcc (orig): {bit_acc_orig.item():.4f}")
        for dist_type, acc in bit_acc_dist.items():
            print(f"   Distortion: {dist_type} | BitAcc: {acc:.4f}")
        log_dict = {
            "psnr": avg_psnr,
            "ssim": avg_ssim,
            "bit_acc_orig": bit_acc_orig.item(),
            "image": wandb.Image(grid_original, caption=f"Epoch {epoch+1}"),
            "wm_image": wandb.Image(grid_watermarked, caption=f"Epoch {epoch+1} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
        }
        # Add bit accuracy for each distortion
        for dist_type, acc in bit_acc_dist.items():
            log_dict[f"bit_acc_{dist_type}"] = acc
        wandb.log(log_dict, step=global_step)


def train_stegastamp(args, accelerator, save_dir):

    #dataset = LAIONDataset("tempertrash/laion_400m", resolution=args.image_resolution)
    
    def preprocess_train(examples):
        # Preprocess images.
        train_transforms = transforms.Compose([
            transforms.Resize((args.image_resolution, args.image_resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        processed_images = [train_transforms(image.convert("RGB")) for image in examples["original_image"]]
        examples["original_pixel_values"] = processed_images

        return examples

    dataset = load_dataset(args.dataset_name, split="train")
    dataset = dataset.with_transform(preprocess_train).select(range(args.max_train_samples))

    def collate_fn(examples):
        original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
        original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
        
        return {
            "original_pixel_values": original_pixel_values,
        }

    test_dataset = load_dataset(args.dataset_name, split="train")
    test_dataset = test_dataset.with_transform(preprocess_train).select(range(args.max_train_samples, args.max_train_samples + args.eval_samples))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=16)

    
    encoder = StegaStampEncoder(
        args.image_resolution,
        3,
        args.watermark_bits,
        return_residual=False
    )
    decoder = StegaStampDecoder(
        args.image_resolution,
        3,
        args.watermark_bits
    )

    encoder, decoder = accelerator.prepare(encoder, decoder)
    encoder, decoder = encoder.to(accelerator.device), decoder.to(accelerator.device)

    optimizer = optim.AdamW(
        params=list(decoder.parameters()) + list(encoder.parameters()), lr=args.learning_rate
    )

    epoch = 0
    max_epochs = args.epochs

    max_steps = max_epochs * ((len(dataset) // args.batch_size) + 1)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * max_steps),  # 或者使用 args.lr_warmup_steps，如果你有固定數值
        num_training_steps=max_steps
    )

    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    global_step = 0
    steps_since_l2_loss_activated = -1

    l2_loss_fn = nn.MSELoss() #lpips.LPIPS(net="vgg").to(accelerator.device)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    activated_epoch = None

    while epoch < max_epochs:
        total_loss = 0.0
        dataloader = DataLoader(
            dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.batch_size, num_workers=16
        )

        progress_bar = tqdm(dataloader, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}/{max_epochs}")

        for _, batch in enumerate(dataloader):
            global_step += 1
            images = batch["original_pixel_values"]

            bs = images.size(0)
            msg = generate_bitstring_watermark(bs, args.watermark_bits)
            
            clean_images, msg = images.to(accelerator.device), msg.to(accelerator.device)

            wm_images = encoder(msg, clean_images)
            wm_images = wm_images * 2 - 1 # -> [-1, 1]
            

            residual = wm_images - clean_images   

            decoder_output = decoder(wm_images)


            l2_loss = l2_loss_fn(wm_images, clean_images)
            bce_loss = bce_loss_fn(decoder_output.reshape(-1), msg.reshape(-1)) # reshape(-1)?

            l2_loss_weight = min(
                max(
                    args.lpips_init_loss_weight,
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
                    print(f"l2 loss activated at Epoch {epoch+1}")
                    steps_since_l2_loss_activated = 0
                    activated_epoch = epoch
            else:
                steps_since_l2_loss_activated += 1
        
            progress_bar.update(1)
            total_loss += loss.item()
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "bit_acc": bitwise_accuracy.item(),
                "step": global_step
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

        if (epoch + 1) % args.save_image_interval == 0:
            evaluate_stegastamp(args, test_dataloader, accelerator, encoder, decoder, save_dir, epoch, global_step)
            
        # if activated_epoch is not None and (epoch - activated_epoch + 1) >= 30:
        #     print(f"Training stopped after 30 epochs post lpips activation at epoch {epoch+1}")
        #     break

        epoch += 1
        
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
