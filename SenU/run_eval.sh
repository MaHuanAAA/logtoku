#!/bin/bash
# eva one pass 
python generate.py --model_name llama2_chat_7B --generate_gt 1  --gpuid 0
python generate.py --model_name llama2_chat_13B --generate_gt 1  --gpuid 0 1
python generate.py --model_name llama2_chat_70B --generate_gt 1  --gpuid 0 1 2 3
python generate.py --model_name llama-3.2-3B-Instruct --generate_gt 1  --gpuid 0
python generate.py --model_name llama-3.1-8B-Instruct --generate_gt 1  --gpuid 0
python generate.py --model_name llama-3.1-70B-Instruct --generate_gt 1  --gpuid 0 1 2 3

# eva muti pass
python muti_eval.py --model 7b --gpuid 0
python muti_eval.py --model 13b --gpuid 0 1
python muti_eval.py --model 70b --gpuid 0 1 2 3
python muti_eval.py --model 3b --gpuid 0
python muti_eval.py --model 8b --gpuid 0
python muti_eval.py --model 70b_3 --gpuid 0 1 2 3
