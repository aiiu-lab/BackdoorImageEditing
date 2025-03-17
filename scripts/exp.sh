#accelerate launch train_watermark_encoder_decoder.py 

#accelerate launch --main_process_port 29501 badedit.py --dataset_name "fusing/instructpix2pix-1000-samples" --backdoor_target_path "/scratch3/users/yufeng/Myproj/static/cat_wo_bg.png"

# accelerate launch --main_process_port 29501 badedit_partial.py --backdoor_rate 0.0 --loss_type "diffusion"
# accelerate launch --main_process_port 29501 badedit_partial.py --backdoor_rate 0.2 --loss_type "diffusion"
# accelerate launch --main_process_port 29501 badedit_partial.py --backdoor_rate 0.3 --loss_type "diffusion"
# accelerate launch --main_process_port 29501 badedit_partial.py --backdoor_rate 0.5 --loss_type "diffusion"

# accelerate launch --main_process_port 29501 badedit_partial.py --backdoor_rate 0.0 --loss_type "combined"
# accelerate launch --main_process_port 29501 badedit_partial.py --backdoor_rate 0.2 --loss_type "combined"
# accelerate launch --main_process_port 29501 badedit_partial.py --backdoor_rate 0.3 --loss_type "combined"
# accelerate launch --main_process_port 29501 badedit_partial.py --backdoor_rate 0.5 --loss_type "combined"


# accelerate launch test_universal_trigger_edit.py --watermark_values 0.3 --backdoor_rate 0.2 --loss_type "diffusion"
# accelerate launch test_universal_trigger_edit.py --watermark_values 0.5 --backdoor_rate 0.2 --loss_type "diffusion"
# accelerate launch test_universal_trigger_edit.py --watermark_values 0.04 --backdoor_rate 0.2 --loss_type "diffusion"

# accelerate launch badedit_partial.py --backdoor_rate 0.2 --loss_type "diffusion"
# accelerate launch badedit_partial.py --backdoor_rate 0.2 --loss_type "combined"
# accelerate launch badedit_partial.py --backdoor_rate 0.1 --loss_type "diffusion"

#accelerate launch badedit_partial.py --backdoor_rate 0.1 --loss_type "combined"
#accelerate launch badedit_partial.py --backdoor_rate 0.2 --loss_type "combined"
accelerate launch badedit_partial.py --backdoor_rate 0.2 --loss_type "diffusion" --train_batch_size 32
accelerate launch badedit_partial.py --backdoor_rate 0.2 --loss_type "combined"


# accelerate launch --main_process_port 29501 badedit.py --dataset_name "fusing/instructpix2pix-1000-samples" --backdoor_target_path "/scratch3/users/yufeng/Myproj/static/pokemon.png"


