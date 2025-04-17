python evaluate_RoSteALS_results_check.py --distortion_type "rotation"
python evaluate_RoSteALS_results_check.py --distortion_type "erasing"
python evaluate_RoSteALS_results_check.py --distortion_type "brightness"
python evaluate_RoSteALS_results_check.py --distortion_type "contrast"   
python evaluate_RoSteALS_results_check.py --distortion_type "compression"
python evaluate_RoSteALS_results_check.py --distortion_type "noise"
python evaluate_RoSteALS_results_check.py --distortion_type "resizedcrop"
python evaluate_RoSteALS_results_check.py --distortion_type "blurring"    

# python evaluate_RoSteALS_results_check.py --distortion_type "clean"


CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --distortion_type "rotation"
CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --distortion_type "erasing"
CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --distortion_type "brightness"
CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --distortion_type "contrast"   
CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --distortion_type "compression"    
CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --distortion_type "noise"
CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --distortion_type "resizedcrop"
CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --distortion_type "blurring"  

# CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --distortion_type "clean"




# DA
# python evaluate_RoSteALS_results_check.py --distortion_type "crop"
# python evaluate_RoSteALS_results_check.py --distortion_type "dropout"
# python evaluate_RoSteALS_results_check.py --distortion_type "hue"
# python evaluate_RoSteALS_results_check.py --distortion_type "saturation"
# python evaluate_RoSteALS_results_check.py --distortion_type "resize"
# python evaluate_RoSteALS_results_check.py --distortion_type "gif"
