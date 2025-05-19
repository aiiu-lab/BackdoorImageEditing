# ablation study (backdoor_rate)
accelerate launch ros_badedit.py --backdoor_target_num 1 --backdoor_rate 0.1 --loss_type "combined"
accelerate launch ros_badedit.py --backdoor_target_num 1 --backdoor_rate 0.2 --loss_type "combined"
accelerate launch ros_badedit.py --backdoor_target_num 1 --backdoor_rate 0.3 --loss_type "combined"
accelerate launch ros_badedit.py --backdoor_target_num 1 --backdoor_rate 0.4 --loss_type "combined"
accelerate launch ros_badedit.py --backdoor_target_num 1 --backdoor_rate 0.5 --loss_type "combined"

accelerate launch stega_badedit.py --backdoor_target_num 1 --backdoor_rate 0.1 --loss_type "combined"
accelerate launch stega_badedit.py --backdoor_target_num 1 --backdoor_rate 0.2 --loss_type "combined"
accelerate launch stega_badedit.py --backdoor_target_num 1 --backdoor_rate 0.3 --loss_type "combined"
accelerate launch stega_badedit.py --backdoor_target_num 1 --backdoor_rate 0.4 --loss_type "combined"
accelerate launch stega_badedit.py --backdoor_target_num 1 --backdoor_rate 0.5 --loss_type "combined"

# ablation study (loss_type)
accelerate launch ros_badedit.py --backdoor_target_num 1 --backdoor_rate 0.1 --loss_type "diffusion"
accelerate launch ros_badedit.py --backdoor_target_num 1 --backdoor_rate 0.1 --loss_type "img"

accelerate launch stega_badedit.py --backdoor_target_num 1 --backdoor_rate 0.1 --loss_type "diffusion"
accelerate launch stega_badedit.py --backdoor_target_num 1 --backdoor_rate 0.1 --loss_type "img"

accelerate launch vine_badedit.py --backdoor_target_num 1 --backdoor_rate 0.1 --loss_type "combined"


# # ablation study (backdoor_target_num)
accelerate launch stega_badedit.py --backdoor_target_num 2 --backdoor_rate 0.1 --loss_type "combined"
accelerate launch stega_badedit.py --backdoor_target_num 4 --backdoor_rate 0.1 --loss_type "combined"

accelerate launch ros_badedit.py --backdoor_target_num 2 --backdoor_rate 0.1 --loss_type "combined"
accelerate launch ros_badedit.py --backdoor_target_num 4 --backdoor_rate 0.1 --loss_type "combined"




