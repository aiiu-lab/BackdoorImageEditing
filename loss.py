# from functools import partial
# from os import terminal_size
# from sched import scheduler

# import torch
# from torch import nn
# import torch.nn.functional as F
    
# def q_sample_diffuser(noise_sched, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor=None) -> torch.Tensor:
#     if noise is None:
#         noise = torch.randn_like(x_start)
        
#     def unqueeze_n(x):
#         return x.reshape(len(x_start), *([1] * len(x_start.shape[1:])))
    
#     alphas_cumprod = noise_sched.alphas_cumprod.to(device=x_start.device, dtype=x_start.dtype)
#     alphas = noise_sched.alphas.to(device=x_start.device, dtype=x_start.dtype)
#     timesteps = timesteps
    
#     sqrt_alphas_cumprod_t = alphas_cumprod[timesteps] ** 0.5
#     sqrt_one_minus_alphas_cumprod_t = (1 - alphas_cumprod[timesteps]) ** 0.5
#     R_coef_t = (1 - alphas[timesteps] ** 0.5) * sqrt_one_minus_alphas_cumprod_t / (1 - alphas[timesteps])
    
#     sqrt_alphas_cumprod_t = unqueeze_n(sqrt_alphas_cumprod_t)
#     R_coef_t = unqueeze_n(R_coef_t)
    
#     noisy_images = noise_sched.add_noise(x_start, noise, timesteps)
    
#     # return sqrt_alphas_cumprod_t * x_start + (1 - sqrt_alphas_cumprod_t) * R + sqrt_one_minus_alphas_cumprod_t * noise, R_coef_t * R + noise 
#     # print(f"x_start shape: {x_start.shape}")
#     # print(f"R shape: {R.shape}")
#     # print(f"timesteps shape: {timesteps.shape}")
#     # print(f"noise shape: {noise.shape}")
#     # print(f"noisy_images shape: {noisy_images.shape}")
#     # print(f"sqrt_alphas_cumprod_t shape: {sqrt_alphas_cumprod_t.shape}")
#     # print(f"R_coef_t shape: {R_coef_t.shape}")
#     return noisy_images + (1 - sqrt_alphas_cumprod_t) * R, R_coef_t * R + noise 

# def p_losses_diffuser(noise_sched, model: nn.Module, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor=None, loss_type: str="l2", encoder_hidden_states=None) -> torch.Tensor:
#     if len(x_start) == 0: 
#         return 0
#     if noise is None:
#         noise = torch.randn_like(x_start)
    
#     x_noisy, target = q_sample_diffuser(noise_sched=noise_sched, x_start=x_start, R=R, timesteps=timesteps, noise=noise)
#     # if clip:
#     #     x_noisy = torch.clamp(x_noisy, min=DEFAULT_VMIN, max=DEFAULT_VMAX)
#     encoder_hidden_states = encoder_hidden_states.repeat(x_noisy.shape[0], 1, 1) 
    
#     predicted_noise = model(x_noisy.contiguous(), timesteps.contiguous(), encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]
    
#     if loss_type == 'l1':
#         loss = F.l1_loss(target, predicted_noise)
#     elif loss_type == 'l2':
#         loss = F.mse_loss(target, predicted_noise)
#     elif loss_type == "huber":
#         loss = F.smooth_l1_loss(target, predicted_noise)
#     else:
#         raise NotImplementedError()

#     return loss



### original version

# %%
from functools import partial
from os import terminal_size
from sched import scheduler

import torch
from torch import nn
import torch.nn.functional as F

def q_sample_diffuser(noise_sched, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor=None) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(x_start)
        
    def unqueeze_n(x):
        return x.reshape(len(x_start), *([1] * len(x_start.shape[1:])))

    alphas_cumprod = noise_sched.alphas_cumprod.to(device=x_start.device, dtype=x_start.dtype)
    alphas = noise_sched.alphas.to(device=x_start.device, dtype=x_start.dtype)
    timesteps = timesteps.to(x_start.device)

    sqrt_alphas_cumprod_t = alphas_cumprod[timesteps] ** 0.5
    sqrt_one_minus_alphas_cumprod_t = (1 - alphas_cumprod[timesteps]) ** 0.5
    R_coef_t = (1 - alphas[timesteps] ** 0.5) * sqrt_one_minus_alphas_cumprod_t / (1 - alphas[timesteps])
    
    sqrt_alphas_cumprod_t = unqueeze_n(sqrt_alphas_cumprod_t)
    R_coef_t = unqueeze_n(R_coef_t)
    
    noisy_images = noise_sched.add_noise(x_start, noise, timesteps)

    # return sqrt_alphas_cumprod_t * x_start + (1 - sqrt_alphas_cumprod_t) * R + sqrt_one_minus_alphas_cumprod_t * noise, R_coef_t * R + noise 
    # print(f"x_start shape: {x_start.shape}")
    # print(f"R shape: {R.shape}")
    # print(f"timesteps shape: {timesteps.shape}")
    # print(f"noise shape: {noise.shape}")
    # print(f"noisy_images shape: {noisy_images.shape}")
    # print(f"sqrt_alphas_cumprod_t shape: {sqrt_alphas_cumprod_t.shape}")
    # print(f"R_coef_t shape: {R_coef_t.shape}")
    return noisy_images + (1 - sqrt_alphas_cumprod_t) * R, R_coef_t * R + noise 

def p_losses_diffuser(noise_sched, model: nn.Module, x_start: torch.Tensor, R: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor=None, loss_type: str="l2") -> torch.Tensor:
    if len(x_start) == 0: 
        return 0
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy, target = q_sample_diffuser(noise_sched=noise_sched, x_start=x_start, R=R, timesteps=timesteps, noise=noise)
    # if clip:
    #     x_noisy = torch.clamp(x_noisy, min=DEFAULT_VMIN, max=DEFAULT_VMAX)
    predicted_noise = model(x_noisy.contiguous(), timesteps.contiguous(), return_dict=False)[0]

    if loss_type == 'l1':
        loss = F.l1_loss(target, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(target, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(target, predicted_noise)
    else:
        raise NotImplementedError()

    return loss



