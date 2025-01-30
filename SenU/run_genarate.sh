#!/bin/bash
python generate.py --model_name llama2_chat_7B --gene 1 --mode one_pass --gpuid 0 
python generate.py --model_name llama2_chat_13B --gene 1 --mode one_pass --gpuid 0 1
python generate.py --model_name llama2_chat_70B --gene 1 --mode one_pass --gpuid 0 1 2 3
python generate.py --model_name llama-3.2-3B-Instruct --gene 1 --mode one_pass --gpuid 0
python generate.py --model_name llama-3.1-8B-Instruct --gene 1 --mode one_pass --gpuid 0
python generate.py --model_name llama-3.1-70B-Instruct --gene 1 --mode one_pass --gpuid 0 1 2 3

python generate.py --model_name llama2_chat_7B --gene 1 --mode muti_pass --temp 0.5 --gpuid 0
python generate.py --model_name llama2_chat_13B --gene 1 --mode muti_pass --temp 0.5 --gpuid 0 1
python generate.py --model_name llama2_chat_70B --gene 1 --mode muti_pass --temp 0.5 --gpuid 0 1 2 3
python generate.py --model_name llama-3.2-3B-Instruct --gene 1 --mode muti_pass --temp 0.5 --gpuid 0
python generate.py --model_name llama-3.1-8B-Instruct --gene 1 --mode muti_pass --temp 0.5 --gpuid 0
python generate.py --model_name llama-3.1-70B-Instruct --gene 1 --mode muti_pass --temp 0.5 --gpuid 0 1 2 3

