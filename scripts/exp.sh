# ablation study (backdoor_rate)
accelerate launch badedit_partial.py --backdoor_target_num 1 --backdoor_rate 0.1 --loss_type "combined"
accelerate launch badedit_partial.py --backdoor_target_num 1 --backdoor_rate 0.2 --loss_type "combined"
accelerate launch badedit_partial.py --backdoor_target_num 1 --backdoor_rate 0.3 --loss_type "combined"
accelerate launch badedit_partial.py --backdoor_target_num 1 --backdoor_rate 0.4 --loss_type "combined"
accelerate launch badedit_partial.py --backdoor_target_num 1 --backdoor_rate 0.5 --loss_type "combined"

accelerate launch stega_badedit_partial.py --backdoor_target_num 1 --backdoor_rate 0.1 --loss_type "combined"
accelerate launch stega_badedit_partial.py --backdoor_target_num 1 --backdoor_rate 0.2 --loss_type "combined"
accelerate launch stega_badedit_partial.py --backdoor_target_num 1 --backdoor_rate 0.3 --loss_type "combined"
accelerate launch stega_badedit_partial.py --backdoor_target_num 1 --backdoor_rate 0.4 --loss_type "combined"
accelerate launch stega_badedit_partial.py --backdoor_target_num 1 --backdoor_rate 0.5 --loss_type "combined"

# ablation study (loss_type)
accelerate launch badedit_partial.py --backdoor_target_num 1 --backdoor_rate 0.1 --loss_type "diffusion"
accelerate launch badedit_partial.py --backdoor_target_num 1 --backdoor_rate 0.1 --loss_type "img"

accelerate launch stega_badedit_partial.py --backdoor_target_num 1 --backdoor_rate 0.1 --loss_type "diffusion"
accelerate launch stega_badedit_partial.py --backdoor_target_num 1 --backdoor_rate 0.1 --loss_type "img"

accelerate launch vine_badedit_partial.py --backdoor_target_num 1 --backdoor_rate 0.1 --loss_type "combined"


# # ablation study (backdoor_target_num)
accelerate launch stega_badedit_partial.py --backdoor_target_num 2 --backdoor_rate 0.1 --loss_type "combined"
accelerate launch stega_badedit_partial.py --backdoor_target_num 4 --backdoor_rate 0.1 --loss_type "combined"

accelerate launch badedit_partial.py --backdoor_target_num 2 --backdoor_rate 0.1 --loss_type "combined"
accelerate launch badedit_partial.py --backdoor_target_num 4 --backdoor_rate 0.1 --loss_type "combined"




# run for paper results
CUDA_VISIBLE_DEVICES=2 python visualize_cleanmodel_results.py
CUDA_VISIBLE_DEVICES=2 python visualize_RoSteALS_results.py
CUDA_VISIBLE_DEVICES=2 python visualize_stega_results.py
CUDA_VISIBLE_DEVICES=2 python visualize_vine_results.py


