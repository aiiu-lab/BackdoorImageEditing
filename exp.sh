#accelerate launch train_watermark_encoder_decoder.py 

#accelerate launch --main_process_port 29501 badedit.py --dataset_name "fusing/instructpix2pix-1000-samples" --backdoor_target_path "/scratch3/users/yufeng/Myproj/static/cat_wo_bg.png"

accelerate launch --main_process_port 29501 badedit_partial.py --dataset_name "fusing/instructpix2pix-1000-samples" --backdoor_target_path "/scratch3/users/yufeng/Myproj/static/cat_wo_bg.png"

# accelerate launch --main_process_port 29501 badedit.py --dataset_name "fusing/instructpix2pix-1000-samples" --backdoor_target_path "/scratch3/users/yufeng/Myproj/static/pokemon.png"


