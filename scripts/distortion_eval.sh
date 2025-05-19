python ./evaluate/evaluate_RoSteALS_results_check.py --distortion_type "rotation"
python ./evaluate/evaluate_RoSteALS_results_check.py --distortion_type "erasing"
python ./evaluate/evaluate_RoSteALS_results_check.py --distortion_type "brightness"
python ./evaluate/evaluate_RoSteALS_results_check.py --distortion_type "contrast"   
python ./evaluate/evaluate_RoSteALS_results_check.py --distortion_type "compression"
python ./evaluate/evaluate_RoSteALS_results_check.py --distortion_type "noise"
python ./evaluate/evaluate_RoSteALS_results_check.py --distortion_type "resizedcrop"
python ./evaluate/evaluate_RoSteALS_results_check.py --distortion_type "blurring"    

# python evaluate_RoSteALS_results_check.py --distortion_type "clean"


python ./evaluate/evaluate_stega_results_check.py --distortion_type "rotation"
python ./evaluate/evaluate_stega_results_check.py --distortion_type "erasing"
python ./evaluate/evaluate_stega_results_check.py --distortion_type "brightness"
python ./evaluate/evaluate_stega_results_check.py --distortion_type "contrast"   
python ./evaluate/evaluate_stega_results_check.py --distortion_type "compression"    
python ./evaluate/evaluate_stega_results_check.py --distortion_type "noise"
python ./evaluate/evaluate_stega_results_check.py --distortion_type "resizedcrop"
python ./evaluate/evaluate_stega_results_check.py --distortion_type "blurring"  

# CUDA_VISIBLE_DEVICES=1 python evaluate_stega_results_check.py --distortion_type "clean"


