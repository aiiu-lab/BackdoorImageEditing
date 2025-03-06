# accelerate launch --main_process_port 29501 main.py --seed 0

accelerate launch train_watermark_encoder_decoder.py 

# accelerate launch badedit.py --dataset_name "diffusers/instructpix2pix-clip-filtered-upscaled"



