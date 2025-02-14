import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
import torchvision.utils as vutils
from torchvision import transforms
import argparse
from datetime import datetime
import random
from typing import Tuple, Union
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup

from modified_stable_diffusion import ModifiedStableDiffusionPipeline, WatermarkGenerator
from loss import p_losses_diffuser
from model import DiffuserModelSched
from dataset import CIFAR10WatermarkedDataset
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional diffusion model with spatial watermark protection")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for the optimizer")
    parser.add_argument("--watermark_bits", type=int, default=10, help="Length of the random bit sequence for watermark")
    parser.add_argument("--alpha", type=float, default=0.05, help="Strength of watermark embedding")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps for training and sampling")
    parser.add_argument("--ckpt", type=str, default="DDPM-CIFAR10-32", help="Pretrained checkpoint")
    parser.add_argument("--clip", type=bool, default=False, help="Whether to clip")
    
    parser.add_argument("--save_image_interval", type=int, default=10, help="Interval to save images during training")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of warmup steps for the learning rate scheduler")
    # parser.add_argument("--logging_dir", type=str, default="./logs", help="Directory for accelerator logs")
    
    parser.add_argument('--save_watermarked_imgs', type=int, default=1, help="Whether to save watermarked images")
    parser.add_argument('--save_root_dir', type=str, default='./runs', help="Root directory to save images")

    parser.add_argument('--test_trigger', type=bool, default=False, help="Whether to use trigger as watermark")
    return parser.parse_args()

def get_box_trig(b1: Tuple[int, int], b2: Tuple[int, int], channel: int, image_size: int, vmin: Union[float, int], vmax: Union[float, int], val: Union[float, int]):
    if isinstance(image_size, int):
        img_shape = (image_size, image_size)
    elif isinstance(image_size, list):
        img_shape = image_size
    else:
        raise TypeError(f"Argument image_size should be either an integer or a list")
    trig = torch.full(size=(channel, *img_shape), fill_value=vmin)

    trig[:, b1[0]:b2[0], b1[1]:b2[1]] = val  
    return trig

def get_trig_mask(trigger: torch.Tensor) -> torch.Tensor:
    """
    Get the mask for the trigger.
    """
    return torch.where(trigger > -1, 0, 1)
    

def embed_watermark(args, labels: torch.Tensor, images: torch.Tensor, watermark: torch.Tensor, alpha: float, is_watermarked: torch.Tensor) -> torch.Tensor:  
    # test whether the box trigger works
    if args.test_trigger:
        trigger = get_box_trig((-16, -16), (-2, -2), 3, 32, -1, 1, 0)
        trigger = trigger.unsqueeze(0).to(images.device)
        mask = get_trig_mask(trigger)
        watermarked_images = images.clone()
        watermarked_images = mask * images + (1-mask) * trigger
        #breakpoint()

        backdoor_watermarked_images = watermarked_images.clone()
        backdoor_watermarked_images[~is_watermarked] = torch.zeros_like(images[~is_watermarked])
        #breakpoint()
        
        return watermarked_images, backdoor_watermarked_images
    # Add watermark to the first channel for watermarked images
    watermarked_images = images.clone()
    watermarked_images = images+ alpha * watermark[labels]

    backdoor_watermarked_images = watermarked_images.clone()
    # Set non-watermarked images to zeros (optional, if needed)
    backdoor_watermarked_images[~is_watermarked] = torch.zeros_like(images[~is_watermarked])
    return watermarked_images , backdoor_watermarked_images

def save_images(images, save_dir, filenames):
    """
    Save images using PIL.Image instead of torchvision.utils.

    Parameters:
    - images: A tensor containing the images to be saved.
    - save_dir: Directory to save the images.
    - filenames: List of filenames to save each image.
    """
    os.makedirs(save_dir, exist_ok=True)

    for i, image in enumerate(images):
        image = transforms.ToPILImage()(image)  
        path = os.path.join(save_dir, filenames[i])
        image.save(path)

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

def sampling(
    args: argparse.Namespace,
    epoch: int,
    dataset: CIFAR10WatermarkedDataset,
    pipeline,
    watermark_patterns: torch.Tensor,
    save_dir: str,
    accelerator: Accelerator,
    num_samples: int = 40
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

    for i in range(num_samples):
        no_w_latents = torch.randn((1, 3, dataset.image_size, dataset.image_size)).to(accelerator.device)

        if args.test_trigger:
            trigger = get_box_trig((-16, -16), (-2, -2), 3, 32, -1, 1, 0)
            trigger = trigger.unsqueeze(0).to(accelerator.device)
            w_latents = no_w_latents.clone()
            w_latents = trigger + no_w_latents
        else:
            # Randomly choose a class from the target_class_list to use as watermark pattern
            random_index = torch.randint(0, len(dataset.target_class_list), (1,)).item()
            selected_class = dataset.target_class_list[random_index]

            selected_watermark = watermark_patterns[selected_class].unsqueeze(0)

            # Embed the watermark into the latent
            w_latents = no_w_latents.clone()
            w_latents = no_w_latents + args.alpha * selected_watermark

        # Generate images using pipeline
        batched_latents = torch.cat([no_w_latents, w_latents], dim=0)

        generated_images = pipeline(
            batch_size=2,  # Change based on requirement
            # generator=torch.manual_seed(42),  # Seed for reproducibility
            init=batched_latents
        ).images  

        clean_image = generated_images[0]
        watermark_image = generated_images[1]
        #clean_image = Image.fromarray((generated_images[0] * 255).astype("uint8"))
        #watermark_image = Image.fromarray((generated_images[1] * 255).astype("uint8"))

        clean_image_path = os.path.join(save_dir, "clean_samples", f"Epoch_{epoch}", f"clean_no_watermark_{i}.png")
        watermark_image_path = os.path.join(save_dir, "backdoor_samples", f"Epoch_{epoch}", f"backdoor_watermark_{i}.png")

        # Save using PIL
        clean_image.save(clean_image_path)
        watermark_image.save(watermark_image_path)




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
    bit_sequences = accelerator.prepare(dataloader.dataset.class_bit_sequences_list).to(accelerator.device)
    watermark_patterns=watermark_generator(bit_sequences).to(accelerator.device) 
    
    cur_step = 0

    for epoch in range(num_epochs):
        progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch + 1}/{num_epochs}")
        
        for _, batch in enumerate(dataloader):
            images, labels, targets, is_watermarked = batch["image"], batch["label"], batch["target"], batch["is_watermarked"]

            # Move data to devices
            images, labels, targets, is_watermarked = accelerator.prepare(images, labels, targets, is_watermarked)
            
            # images_latent = vae.encode(images).latent_dist.sample() * 0.18215  
            # targets_latent = vae.encode(targets).latent_dist.sample() * 0.18215
            images , labels, targets, is_watermarked = images.to(accelerator.device), labels.to(accelerator.device), targets.to(accelerator.device), is_watermarked.to(accelerator.device)  # not sure if this is necessary
            #breakpoint()
            watermarked_images, backdoor_watermarked_images = embed_watermark(args, labels, images, watermark_patterns, args.alpha, is_watermarked=is_watermarked) 
            #breakpoint()
            # Add noise to latents
            backdoor_watermarked_images = backdoor_watermarked_images.detach()
            noise = torch.randn_like(images).to(accelerator.device)
            timesteps = torch.randint(0, noise_sched.config.num_train_timesteps, (images.size(0),),device=accelerator.device).long()
            # timesteps = torch.Tensor([args.num_inference_steps]).repeat(images.size(0)).long().to(accelerator.device)
            
            with accelerator.accumulate(model):
                # Compute losses
                optimizer.zero_grad()
                loss_diffuser = p_losses_diffuser(
                    noise_sched=noise_sched,
                    model=model,
                    x_start=targets,
                    R=backdoor_watermarked_images,
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


            # save img in the last epoch
            if args.save_watermarked_imgs and epoch == num_epochs - 1:

                save_images(
                    images,
                    save_dir=os.path.join(save_dir, "no_watermarked"),
                    filenames=[f"no_{i}.png" for i in range(len(images))]
                )

                save_images(
                    watermarked_images,
                    save_dir=os.path.join(save_dir, "watermarked"),
                    filenames=[f"w_{i}.png" for i in range(len(watermarked_images))]
                )

        if  accelerator.is_main_process:
            pipeline = get_pipeline(unet= accelerator.unwrap_model(model), scheduler= noise_sched)
            # Unwrap the model (add to check if it works)
            # pipeline.unet = accelerator.unwrap_model(pipeline.unet)
            if(epoch + 1) % args.save_image_interval == 0 or epoch == num_epochs - 1:
                # run inference and save images
                sampling(args, epoch, dataloader.dataset, pipeline, watermark_patterns, save_dir, accelerator)
            if epoch == num_epochs - 1:
                # Save the last checkpoint
                checkpoint(accelerator, pipeline, epoch, save_dir)
                



def main():
    args = parse_args()

    print("Use Trigger:", args.test_trigger)

    # Initialize Accelerator
    accelerator = Accelerator()

    # Load dataset
    dataset = CIFAR10WatermarkedDataset(
        root="./data",
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
