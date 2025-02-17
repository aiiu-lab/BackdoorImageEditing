import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
import torchvision.utils as vutils
from torchvision import transforms
import argparse
import wandb
import numpy as np
from datetime import datetime
import random
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup

from modified_stable_diffusion import ModifiedStableDiffusionPipeline, WatermarkGenerator
from loss import p_losses_diffuser
from model import DiffuserModelSched
from dataset import CIFAR10WatermarkedDataset
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional diffusion model with spatial watermark protection")
    parser.add_argument("--project", type=str, default="Watermark_Baddiffusion", help="Project name for wandb")

    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for the optimizer")
    parser.add_argument("--watermark_bits", type=int, default=10, help="Length of the random bit sequence for watermark")
    parser.add_argument("--alpha", type=float, default=0.05, help="Strength of watermark embedding")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps for training and sampling")
    parser.add_argument("--ckpt", type=str, default="DDPM-CIFAR10-32", help="Pretrained checkpoint")
    parser.add_argument("--clip", type=bool, default=False, help="Whether to clip")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--mixed_precision", type=str, default="fp16", help="Mixed precision mode")
    
    parser.add_argument("--save_image_interval", type=int, default=10, help="Interval to save images during training")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps for the learning rate scheduler")
    # parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for accelerator logs")
    
    parser.add_argument('--save_watermarked_imgs', type=int, default=1, help="Whether to save watermarked images")
    parser.add_argument('--save_root_dir', type=str, default='./runs', help="Root directory to save images")
    parser.add_argument('--log_type', type=str, default='wandb', help="Logging type")

    parser.add_argument('--test_trigger', type=bool, default=False, help="Whether to use trigger as watermark")
    parser.add_argument('--debug_mode', type=bool, default=False, help="Whether to use debug mode")
    return parser.parse_args()
    

def embed_watermark(args, labels: torch.Tensor, images: torch.Tensor, watermark: torch.Tensor, alpha: float, is_watermarked: torch.Tensor) -> torch.Tensor:  
    # test whether the box trigger works

    # if args.test_trigger:
    #     trigger = get_box_trig((-16, -16), (-2, -2), 3, 32, -1, 1, 0)
    #     trigger = trigger.unsqueeze(0).to(images.device)
    #     mask = get_trig_mask(trigger)
    #     watermarked_images = images.clone()
    #     watermarked_images = mask * images + (1-mask) * trigger
    #     #breakpoint()

    #     backdoor_watermarked_images = watermarked_images.clone()
    #     backdoor_watermarked_images[~is_watermarked] = torch.zeros_like(images[~is_watermarked])
    #     #breakpoint()
        
    #     return watermarked_images, backdoor_watermarked_images
    # Add watermark to the first channel for watermarked images
    watermarked_images = images.clone()
    watermarked_images = images+ alpha * watermark[labels]

    backdoor_watermarked_images = watermarked_images.clone()
    # Set non-watermarked images to zeros (optional, if needed)
    backdoor_watermarked_images[~is_watermarked] = torch.zeros_like(images[~is_watermarked])
    return watermarked_images , backdoor_watermarked_images

def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    將 tensor 從 [-1, 1] 反歸一化到 [0, 1] 範圍。
    """
    return (tensor + 1) / 2

def save_images(images: torch.Tensor, pixel_values: torch.Tensor, is_watermarked: torch.Tensor, save_dir: str):
    """
    當 is_watermarked 為 True 時，
    將原始圖片 (images) 儲存到 no_watermark 資料夾，
    將反歸一化後的 pixel_values 儲存到 watermarked 資料夾。
    
    參數：
    - images: 原始圖片 tensor，形狀為 [B, C, H, W]。
    - pixel_values: 加 watermark 後圖片 tensor，形狀為 [B, C, H, W]（範圍 [-1, 1]）。
    - is_watermarked: 布林值 tensor，形狀為 [B]，指示每個樣本是否被加 watermark。
    - save_dir: 儲存圖片的根目錄。
    """
    # 建立存檔資料夾
    no_wm_folder = os.path.join(save_dir, "no_watermark")
    wm_folder = os.path.join(save_dir, "watermarked")
    os.makedirs(no_wm_folder, exist_ok=True)
    os.makedirs(wm_folder, exist_ok=True)

    # 定義 PIL 轉換器
    to_pil = transforms.ToPILImage()

    # 遍歷 batch 中的每個樣本
    for idx in range(images.shape[0]):
        if is_watermarked[idx]:
            im_tensor = denormalize(images[idx].cpu())
            orig_img = to_pil(im_tensor)
            orig_path = os.path.join(no_wm_folder, f"image_{idx}.png")
            orig_img.save(orig_path)

            # 對 pixel_values 進行反歸一化後儲存
            wm_tensor = denormalize(pixel_values[idx].cpu())
            wm_img = to_pil(wm_tensor)
            wm_path = os.path.join(wm_folder, f"image_{idx}.png")
            wm_img.save(wm_path)

def checkpoint(
    accelerator: Accelerator,
    pipeline,
    cur_epoch: int,
    output_dir: str
):
    """
    Save checkpoint using the accelerator and save model state.

    Parameters:
    - accelerator: Accelerator, the distributed training accelerator.
    - pipeline: Model pipeline (e.g., Stable Diffusion pipeline).
    - cur_epoch: int, the current epoch.
    - output_dir: str, directory to save checkpoints.
    """

    # Prepare checkpoint directory
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save state using accelerator
    accelerator.save_state(os.path.join(ckpt_dir, f"accelerator_state_epoch_{cur_epoch}.pt"))

    # Save additional metadata
    metadata = {
        "epoch": cur_epoch,
    }
    torch.save(metadata, os.path.join(ckpt_dir, f"training_state_epoch_{cur_epoch}.pt"))

    # Save the model
    model_save_path = os.path.join(output_dir, f"models")
    os.makedirs(model_save_path, exist_ok=True)
    pipeline.save_pretrained(model_save_path)

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def sampling(
    args: argparse.Namespace,
    epoch: int,
    dataset: CIFAR10WatermarkedDataset,
    pipeline,
    watermark_pattern: torch.Tensor,
    save_dir: str,
    accelerator: Accelerator,
    num_samples: int = 25
):
    """
    Perform inference on the trained pipeline with watermarking.

    Parameters:
    - args: argparse.Namespace, the input arguments.
    - img_size: int, the size of the generated images.
    - pipeline: ModifiedStableDiffusionPipeline, the trained pipeline.
    - watermark_patterns: torch.Tensor, the generated watermark patterns.
    - save_dir: str, the directory to save generated images.
    - accelerator: Accelerator, the distributed training accelerator.
    - num_samples: int, the number of samples to generate
    """
    # Generate random latent space samples

    os.makedirs(os.path.join(save_dir, "clean_samples", f"Epoch_{epoch}"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "backdoor_samples", f"Epoch_{epoch}"), exist_ok=True)

    with torch.no_grad():
        no_w_latents = torch.randn((num_samples, 3, dataset.image_size, dataset.image_size),
                                   generator=torch.manual_seed(args.seed)).to(accelerator.device)

        # if args.test_trigger:
        #     trigger = get_box_trig((-16, -16), (-2, -2), 3, 32, -1, 1, 0)
        #     trigger = trigger.unsqueeze(0).to(accelerator.device)
        #     w_latents = no_w_latents.clone()
        #     w_latents = trigger + no_w_latents
        # else:
        #     # Randomly choose a class from the target_class_list to use as watermark pattern
        #     # random_index = torch.randint(0, len(dataset.target_class_list), (1,)).item()
        #     # selected_class = dataset.target_class_list[random_index]

        #     # selected_watermark = watermark_patterns[selected_class].unsqueeze(0)

        #     # Embed the watermark into the latent
        #     w_latents = no_w_latents.clone()
        #     w_latents = no_w_latents + watermark_pattern  
        #     w_latents = torch.clamp(w_latents, -1.0, 1.0)

        # Generate images using pipeline
        #batched_latents = torch.cat([no_w_latents, w_latents], dim=0)

        generated_images = pipeline(
            batch_size=num_samples,  
            generator=torch.manual_seed(args.seed),  
            init=no_w_latents,
        ).images  
        
        #images = [Image.fromarray(image) for image in np.squeeze((generated_images * 255).round().astype("uint8"))]
        clean_image_grid = make_grid(generated_images, 5, 5)

        w_latents = no_w_latents.clone()
        w_latents = no_w_latents + watermark_pattern  
        # w_latents = torch.clamp(w_latents, -1.0, 1.0) # this decreases performance


        generated_images = pipeline(
            batch_size=num_samples,  
            generator=torch.manual_seed(args.seed),  
            init=w_latents,
        ).images  

        #images = [Image.fromarray(image) for image in np.squeeze((generated_images * 255).round().astype("uint8"))]
        backdoor_image_grid = make_grid(generated_images, 5, 5)

        clean_image_grid.save(os.path.join(save_dir, "clean_samples", f"Epoch_{epoch}", f"clean_samples_{epoch}.png"))
        backdoor_image_grid.save(os.path.join(save_dir, "backdoor_samples", f"Epoch_{epoch}", f"backdoor_samples_{epoch}.png"))


        




def train_unet_with_watermark(
    args: argparse.Namespace,
    model: UNet2DModel,
    noise_sched,
    get_pipeline, # function to get pipeline
    dataloader: DataLoader,
    watermark_generator: WatermarkGenerator,
    optimizer: torch.optim.Optimizer,
    lr_sched: torch.optim.lr_scheduler.LambdaLR,
    num_epochs: int,
    accelerator: Accelerator,
):
    
    if not args.debug_mode:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = os.path.join(args.save_root_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)

    # unet = pipeline.unet
    # unet.train()

    # noise_scheduler = pipeline.scheduler
    # vae = pipeline.vae
    # text_input=""
    # text_inputs = pipeline.tokenizer(
    #     text_input,
    #     padding="max_length",
    #     max_length=pipeline.tokenizer.model_max_length,
    #     truncation=True,
    #     return_tensors="pt"
    # ).to(accelerator.device)

    # encoder_hidden_states = pipeline.text_encoder(
    #     input_ids=text_inputs["input_ids"],
    #     attention_mask=text_inputs["attention_mask"]
    # ).last_hidden_state.to(accelerator.device)
    pipeline = get_pipeline(unet= accelerator.unwrap_model(model), scheduler= noise_sched)  # get pipeline
    pipeline.to(accelerator.device)
    # Generate watermark patterns
    #bit_sequences = accelerator.prepare(dataloader.dataset.class_bit_sequences_list).to(accelerator.device)
    #watermark_patterns=watermark_generator(bit_sequences).to(accelerator.device) 
    watermark_pattern = dataloader.dataset.watermark_pattern.to(accelerator.device)
    
    cur_step = 0

    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
        
        for _, batch in enumerate(dataloader):
            pixel_values, labels, targets, is_watermarked = batch["pixel_values"], batch["label"], batch["target"], batch["is_watermarked"]

            # Move data to devices
            pixel_values, labels, targets, is_watermarked = accelerator.prepare(pixel_values, labels, targets, is_watermarked)
            # images_latent = vae.encode(images).latent_dist.sample() * 0.18215  
            # targets_latent = vae.encode(targets).latent_dist.sample() * 0.18215
            pixel_values , labels, targets, is_watermarked = pixel_values.to(accelerator.device), labels.to(accelerator.device), targets.to(accelerator.device), is_watermarked.to(accelerator.device)  # not sure if this is necessary
            #watermarked_images, backdoor_watermarked_images = embed_watermark(args, labels, images, watermark_patterns, args.alpha, is_watermarked=is_watermarked) 
            #breakpoint()
            # Add noise to latents
            #backdoor_watermarked_images = backdoor_watermarked_images.detach()
            noise = torch.randn_like(pixel_values).to(accelerator.device)
            timesteps = torch.randint(0, noise_sched.config.num_train_timesteps, (pixel_values.size(0),),device=accelerator.device).long()
            # timesteps = torch.Tensor([args.num_inference_steps]).repeat(images.size(0)).long().to(accelerator.device)

            with accelerator.accumulate(model):
                # Compute losses
                optimizer.zero_grad()
                loss_diffuser = p_losses_diffuser(
                    noise_sched=noise_sched,
                    model=model,
                    x_start=targets,
                    R=pixel_values,
                    timesteps=timesteps,
                    noise=noise,
                    loss_type="l2",
                )

                accelerator.backward(loss_diffuser)  # retain_graph=True

                # watermarked_images = vae.decode(watermarked_latents / 0.18215).sample.to(accelerator.device)
                
                # Compute MSE loss
                #loss_mse = mse_loss(
                #    watermarked_images,
                #    images
                #)

                # mse loss update watermark_generator
                #optimizer_wm.zero_grad()
                #accelerator.backward(loss_mse, retain_graph=True)
                #optimizer_wm.step()
                #scheduler_wm.step()

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                # diffuser loss update unet
                optimizer.step()
                lr_sched.step()
            
            cur_step += 1
                

            progress_bar.update(1)
            logs = {
                "loss": loss_diffuser.detach().item(),
                "lr": lr_sched.get_last_lr()[0],
                "step": cur_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=cur_step)


            # save img in the last epoch
            if args.save_watermarked_imgs and epoch == num_epochs - 1:

                save_images(
                    batch["image"],
                    pixel_values,
                    is_watermarked,
                    save_dir=save_dir,
                )

        if  accelerator.is_main_process:
            pipeline = get_pipeline(unet= accelerator.unwrap_model(model), scheduler= noise_sched)
            # Unwrap the model (add to check if it works)
            # pipeline.unet = accelerator.unwrap_model(pipeline.unet)
            if(epoch + 1) % args.save_image_interval == 0 or epoch == num_epochs - 1:
                # run inference and save images
                sampling(args, epoch, dataloader.dataset, pipeline, watermark_pattern, save_dir, accelerator)
            if epoch == num_epochs - 1:
                # Save the last checkpoint
                checkpoint(accelerator, pipeline, epoch, save_dir)
                



def main():
    args = parse_args()

    # Initialize Accelerator
    accelerator = Accelerator(
        log_with=args.log_type,
        mixed_precision=args.mixed_precision
    )

    if accelerator.is_main_process:
        print("Use Trigger:", args.test_trigger)
        print("seed:", args.seed)
        wandb.init(project=args.project, name=args.ckpt, id=args.ckpt, config=vars(args))
        accelerator.init_trackers(args.project, config=vars(args))


    # Load dataset
    dataset = CIFAR10WatermarkedDataset(
        args = args,
        name="CIFAR10",
        train=True,
        bit_length=args.watermark_bits,
        image_size=32,
        target_class_list=[0,1,2],
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # # Initialize pipeline
    # pipeline = ModifiedStableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
    # pipeline.to(accelerator.device)

    model, noise_sched, get_pipeline = DiffuserModelSched.get_pretrained(ckpt=args.ckpt, clip_sample=args.clip)

    # Initialize watermark generator
    watermark_generator = WatermarkGenerator(bit_length=args.watermark_bits, image_size=32, channel=1)

    # Define optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_sched = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(dataloader) * args.epochs),
    )

    # Prepare everything with accelerator
    model, noise_sched, watermark_generator, optimizer,  lr_sched = accelerator.prepare(
        model, noise_sched, watermark_generator, optimizer, lr_sched
    )

    # Train the model
    train_unet_with_watermark(
        args=args,
        model=model,
        noise_sched=noise_sched,
        get_pipeline=get_pipeline,
        dataloader=dataloader,
        watermark_generator=watermark_generator,
        optimizer=optimizer,
        lr_sched=lr_sched,
        num_epochs=args.epochs,
        accelerator=accelerator,
    )


if __name__ == "__main__":
    main()
