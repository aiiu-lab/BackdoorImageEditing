# rosteals
python evaluate_RoSteALS_results_check.py --ckpt_path "./instruct-pix2pix-model/runs/Multi1_RoSteALS_bad_0.1_combined_2025-04-02_11-32/ckpt_epoch_48"
python evaluate_RoSteALS_results_check.py --ckpt_path "./instruct-pix2pix-model/runs/Multi1_RoSteALS_bad_0.2_combined_2025-04-01_09-41/ckpt_epoch_40"
python evaluate_RoSteALS_results_check.py --ckpt_path "./instruct-pix2pix-model/runs/Multi1_RoSteALS_bad_0.3_combined_2025-04-02_19-09/ckpt_epoch_47" # check
python evaluate_RoSteALS_results_check.py --ckpt_path "./instruct-pix2pix-model/runs/Multi1_RoSteALS_bad_0.4_combined_2025-04-03_02-45/ckpt_epoch_46"
python evaluate_RoSteALS_results_check.py --ckpt_path "./instruct-pix2pix-model/runs/Multi1_RoSteALS_bad_0.5_combined_2025-04-03_10-20/ckpt_epoch_48"

# ablation study (loss_type)
python evaluate_RoSteALS_results_check.py --ckpt_path "./instruct-pix2pix-model/runs/Multi1_RoSteALS_bad_0.1_diffusion_2025-04-04_23-46/ckpt_epoch_42"
python evaluate_RoSteALS_results_check.py --ckpt_path "./instruct-pix2pix-model/runs/Multi1_RoSteALS_bad_0.1_img_2025-04-05_06-02/ckpt_epoch_44"

# ablation study (backdoor_target_num)
python evaluate_RoSteALS_results_check.py --backdoor_target_num 2 --ckpt_path "./instruct-pix2pix-model/runs/Multi2_RoSteALS_bad_0.1_combined_2025-04-06_22-37/ckpt_epoch_42"
python evaluate_RoSteALS_results_check.py --backdoor_target_num 4 --ckpt_path "./instruct-pix2pix-model/runs/Multi4_RoSteALS_bad_0.1_combined_2025-04-07_06-13/ckpt_epoch_45"

# stega
CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --ckpt_path "./stega_instruct-pix2pix-model/runs/Multi1_Stega_bad_0.1_combined_2025-04-03_17-57/ckpt_epoch_44"
CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --ckpt_path "./stega_instruct-pix2pix-model/runs/Multi1_Stega_bad_0.2_combined_2025-04-01_17-17/ckpt_epoch_39"
CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --ckpt_path "./stega_instruct-pix2pix-model/runs/Multi1_Stega_bad_0.3_combined_2025-04-03_23-41/ckpt_epoch_47"
CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --ckpt_path "./stega_instruct-pix2pix-model/runs/Multi1_Stega_bad_0.4_combined_2025-04-04_05-26/ckpt_epoch_49"
CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --ckpt_path "./stega_instruct-pix2pix-model/runs/Multi1_Stega_bad_0.5_combined_2025-04-04_11-11/ckpt_epoch_49"

# ablation study (loss_type)
CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --ckpt_path "./stega_instruct-pix2pix-model/runs/Multi1_Stega_bad_0.1_diffusion_2025-04-05_13-39/ckpt_epoch_47"
CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --ckpt_path "./stega_instruct-pix2pix-model/runs/Multi1_Stega_bad_0.1_img_2025-04-05_18-02/ckpt_epoch_48"

# ablation study (backdoor_target_num)
CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --backdoor_target_num 2 --ckpt_path "/scratch3/users/yufeng/Myproj/stega_instruct-pix2pix-model/runs/Multi2_Stega_bad_0.1_combined_2025-04-06_11-07/ckpt_epoch_35"
CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --backdoor_target_num 4 --ckpt_path "/scratch3/users/yufeng/Myproj/stega_instruct-pix2pix-model/runs/Multi4_Stega_bad_0.1_combined_2025-04-06_16-52/ckpt_epoch_47"

# VINE
CUDA_VISIBLE_DEVICES=2 python evaluate_vine_results_check.py --ckpt_path "./VINE-B_instruct-pix2pix-model/runs/Multi1_VINE_bad_0.1_combined_2025-04-05_23-47/ckpt_epoch_37"

